"""租约 fencing 回归测试。

危险窗口：工作进程 A 在租约有效时开始计算，结果尚未写回时租约到期并被 B
接管。旧实现允许 A 随后仍然写入分析、推进批次、完成任务并留下成功审计。

这里用可注入的冻结时钟和受控的领取顺序复现该窗口（全程无需真实等待），并断言：
失去租约的 A 即使已经算出结果，也不会留下任何成功痕迹；B 完成后任务、分析、
批次和成功审计都只出现一次；合法持有人对同一输入快照重放得到幂等结果。
"""

from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from pathlib import Path

from robot_trials.analysis import analyze
from robot_trials.clock import FrozenClock
from robot_trials.errors import InvalidState
from robot_trials.jsonio import load_json
from robot_trials.service import TrialService


ROOT = Path(__file__).resolve().parents[1]


class FencingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = TrialService(self.connection, self.clock)
        for user_id, role in (
            ("operator", "operator"),
            ("stat", "statistician"),
            ("approver", "approver"),
            ("auditor", "auditor"),
        ):
            self.service.create_user(user_id, user_id, role)
        protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.protocol = protocol
        self.service.register_robot("operator", "robot-a", "A 型", "厂商")
        self.service.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        self.service.publish_protocol("stat", protocol)
        self.service.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        self.service.start_batch("operator", "batch-a", 1)
        self.service.import_observations("operator", "batch-a", "key-1", rows)
        self.service.seal_batch("stat", "batch-a", 2)

    def tearDown(self) -> None:
        self.connection.close()

    def _scalar(self, sql: str, *params: object) -> int:
        return self.connection.execute(sql, params).fetchone()[0]

    def test_stale_worker_is_rejected_and_new_worker_completes_once(self) -> None:
        # t0：A 领取任务，拿到第 1 代持有凭证并开始计算。
        claim_a = self.service.claim_job("worker-a", lease_seconds=10)
        self.assertIsNotNone(claim_a)
        job_id = claim_a["job_id"]
        token_a = claim_a["lease_generation"]
        self.assertEqual(token_a, 1)

        # A 在租约有效期内已经把结果算了出来（结果只存在于进程内存里）。
        batch = self.service.get_batch("batch-a")
        protocol, _ = self.service._protocol(batch["protocol_id"], batch["protocol_version"])
        result_a = analyze(protocol, self.service._analysis_observations("batch-a", protocol))
        self.assertEqual(result_a["conclusion"], "pass")

        # 计算尚未写回，租约到期：时钟推进，无需真实等待。
        self.clock.advance(seconds=11)

        # B 按受控顺序接管同一任务，获得第 2 代凭证；A 手里的凭证立即失效。
        claim_b = self.service.claim_job("worker-b", lease_seconds=10)
        self.assertEqual(claim_b["job_id"], job_id)
        self.assertEqual(claim_b["lease_owner"], "worker-b")
        token_b = claim_b["lease_generation"]
        self.assertGreater(token_b, token_a)

        # A 在写回时出示旧凭证（即使它确实算出了结果）：必须被拒绝。
        with self.assertRaises(InvalidState) as caught:
            self.service.complete_job("worker-a", job_id, "stat", token_a)
        self.assertIn("代次", str(caught.exception))

        # 拒绝之后没有任何成功痕迹：无分析、批次仍封存、任务仍由 B 持有、无成功审计。
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 0)
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses"), 0)
        job_row = self.connection.execute("SELECT * FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        self.assertEqual(job_row["state"], "leased")
        self.assertEqual(job_row["lease_owner"], "worker-b")
        self.assertEqual(job_row["lease_generation"], token_b)
        self.assertEqual(
            self._scalar(
                "SELECT count(*) FROM audit_events WHERE event_type='analysis.completed'"
            ),
            0,
        )
        batch_row = self.connection.execute("SELECT state FROM batches WHERE batch_id='batch-a'").fetchone()
        self.assertEqual(batch_row["state"], "sealed")
        # 拒绝本身留痕，但属于拒绝台账，不是成功审计。
        self.assertEqual(
            self._scalar(
                "SELECT count(*) FROM job_events WHERE job_id=? AND event_type='complete_rejected'",
                job_id,
            ),
            1,
        )
        rejection = self.connection.execute(
            "SELECT worker_id,lease_generation FROM job_events WHERE event_type='complete_rejected'"
        ).fetchone()
        self.assertEqual(rejection["worker_id"], "worker-a")
        self.assertEqual(rejection["lease_generation"], token_a)

        # B 作为合法持有人完成：分析、任务、批次、成功审计都恰好一次。
        completed_b = self.service.complete_job("worker-b", job_id, "stat", token_b)
        self.assertEqual(completed_b["result"]["conclusion"], "pass")
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses"), 1)
        self.assertEqual(
            self._scalar("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 1
        )
        job_final = self.connection.execute("SELECT * FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        self.assertEqual(job_final["state"], "succeeded")
        self.assertIsNone(job_final["lease_owner"])
        self.assertEqual(
            self.connection.execute("SELECT state FROM batches WHERE batch_id='batch-a'").fetchone()["state"],
            "analyzed",
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM audit_events WHERE event_type='analysis.completed'"),
            1,
        )
        self.assertEqual(
            self._scalar("SELECT count(*) FROM job_events WHERE job_id=? AND event_type='succeeded'", job_id),
            1,
        )

        # A 在 B 已完成之后再拿旧凭证补写：仍然被拒绝，且记录数量不发生变化。
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-a", job_id, "stat", token_a)
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses"), 1)
        self.assertEqual(job_final["state"], "succeeded")
        self.assertEqual(
            self._scalar("SELECT count(*) FROM audit_events WHERE event_type='analysis.completed'"),
            1,
        )

    def test_legitimate_holder_replay_is_idempotent_on_same_snapshot(self) -> None:
        claim = self.service.claim_job("worker-b", lease_seconds=10)
        token = claim["lease_generation"]
        first = self.service.complete_job("worker-b", claim["job_id"], "stat", token)

        # 合法持有人对同一输入快照重放完成请求：返回同一分析，不新增任何记录。
        replay = self.service.complete_job("worker-b", claim["job_id"], "stat", token)
        self.assertEqual(replay["analysis_id"], first["analysis_id"])
        self.assertEqual(replay["input_sha256"], first["input_sha256"])
        self.assertEqual(replay["result"], first["result"])
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses"), 1)
        self.assertEqual(
            self._scalar("SELECT count(*) FROM audit_events WHERE event_type='analysis.completed'"),
            1,
        )
        self.assertEqual(
            self._scalar(
                "SELECT count(*) FROM job_events WHERE event_type='complete_rejected'"
            ),
            0,
        )

    def test_wrong_worker_presenting_live_token_is_rejected(self) -> None:
        claim = self.service.claim_job("worker-b", lease_seconds=10)
        # 另一工作进程冒用同一代凭证：持有人不匹配，整事务回滚。
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-a", claim["job_id"], "stat", claim["lease_generation"])
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses"), 0)
        self.assertEqual(
            self.connection.execute(
                "SELECT state,lease_owner FROM analysis_jobs WHERE job_id=?", (claim["job_id"],)
            ).fetchone()["state"],
            "leased",
        )

    def test_forged_generation_is_rejected(self) -> None:
        claim = self.service.claim_job("worker-b", lease_seconds=10)
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-b", claim["job_id"], "stat", claim["lease_generation"] + 999)
        self.assertEqual(self._scalar("SELECT count(*) FROM analyses"), 0)

    def test_stale_worker_cannot_release_lease_either(self) -> None:
        # A 的租约过期被 B 接管后，A 连"主动失败/释放租约"也必须被拒绝。
        claim_a = self.service.claim_job("worker-a", lease_seconds=10)
        self.clock.advance(seconds=11)
        claim_b = self.service.claim_job("worker-b", lease_seconds=10)
        with self.assertRaises(InvalidState):
            self.service.fail_job("worker-a", claim_a["job_id"], "A 迟来的失败上报",
                                  lease_generation=claim_a["lease_generation"])
        job_row = self.connection.execute(
            "SELECT state,lease_owner,lease_generation,last_error FROM analysis_jobs WHERE job_id=?",
            (claim_b["job_id"],),
        ).fetchone()
        self.assertEqual(job_row["state"], "leased")
        self.assertEqual(job_row["lease_owner"], "worker-b")
        self.assertEqual(job_row["lease_generation"], claim_b["lease_generation"])
        self.assertIsNone(job_row["last_error"])


if __name__ == "__main__":
    unittest.main()
