from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from pathlib import Path

from robot_trials.clock import FrozenClock
from robot_trials.errors import Conflict, Forbidden, InvalidState
from robot_trials.jsonio import load_json
from robot_trials.service import TrialService


ROOT = Path(__file__).resolve().parents[1]


class ServiceTests(unittest.TestCase):
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
        self.protocol = load_json(ROOT / "fixtures" / "demo_protocol.json")
        self.rows = [
            json.loads(line)
            for line in (ROOT / "fixtures" / "demo_observations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.service.register_robot("operator", "robot-a", "A 型", "厂商")
        self.service.register_build("operator", "build-a", "robot-a", "1.0", "b" * 64)
        self.service.publish_protocol("stat", self.protocol)
        self.service.create_batch("operator", "batch-a", "demo-delivery-v1", 1, "build-a")
        self.service.start_batch("operator", "batch-a", 1)

    def tearDown(self) -> None:
        self.connection.close()

    def test_complete_workflow(self) -> None:
        imported = self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.assertEqual(imported["inserted"], 6)
        self.service.seal_batch("stat", "batch-a", 2)
        job = self.service.claim_job("worker", 30)
        analysis = self.service.complete_job("worker", job["job_id"], "stat", job["claim_generation"])
        self.service.decide("approver", "batch-a", analysis["analysis_id"], "approved", "满足规则")
        report = self.service.report("auditor", "batch-a")
        self.assertEqual(report["batch"]["state"], "decided")
        self.assertEqual(report["analysis"]["result"]["conclusion"], "pass")

    def test_idempotent_replay_and_conflict(self) -> None:
        first = self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        second = self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.assertEqual(first, second)
        changed = [dict(item) for item in self.rows]
        changed[0] = dict(changed[0])
        changed[0]["metrics"] = dict(changed[0]["metrics"])
        changed[0]["metrics"]["completion_seconds"] = "99"
        with self.assertRaises(Conflict):
            self.service.import_observations("operator", "batch-a", "key-1", changed)
        count = self.connection.execute("SELECT count(*) FROM observations").fetchone()[0]
        self.assertEqual(count, 6)

    def test_import_rolls_back_when_one_source_row_duplicates(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows[:1])
        with self.assertRaises(Conflict):
            self.service.import_observations("operator", "batch-a", "key-2", self.rows[:2])
        count = self.connection.execute("SELECT count(*) FROM observations").fetchone()[0]
        self.assertEqual(count, 1)

    def test_role_separation(self) -> None:
        with self.assertRaises(Forbidden):
            self.service.seal_batch("operator", "batch-a", 2)
        with self.assertRaises(Forbidden):
            self.service.report("operator", "batch-a")

    def test_exclusion_review_and_revoke_leave_history(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        observation_id = self.connection.execute(
            "SELECT observation_id FROM observations ORDER BY observation_id LIMIT 1"
        ).fetchone()[0]
        requested = self.service.request_exclusion("operator", observation_id, "现场记录失效")
        reviewed = self.service.review_exclusion("stat", requested["exclusion_id"], True, "证据充分")
        self.assertEqual(reviewed["status"], "approved")
        revoked = self.service.revoke_exclusion("operator", requested["exclusion_id"], "已找回原始记录")
        self.assertEqual(revoked["status"], "revoked")
        events = self.connection.execute(
            "SELECT event_type FROM audit_events WHERE entity_type='observation' AND entity_id=? ORDER BY event_id",
            (str(observation_id),),
        ).fetchall()
        self.assertEqual([row[0] for row in events], ["exclusion.requested", "exclusion.revoked"])

    def test_failed_job_returns_to_queue_after_delay(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.service.seal_batch("stat", "batch-a", 2)
        job = self.service.claim_job("worker-a", 10)
        failed = self.service.fail_job(
            "worker-a", job["job_id"], "临时计算失败", job["claim_generation"], retry_seconds=5
        )
        self.assertEqual(failed["state"], "queued")
        self.assertIsNone(self.service.claim_job("worker-b", 10))
        self.clock.advance(seconds=5)
        retried = self.service.claim_job("worker-b", 10)
        self.assertEqual(retried["job_id"], job["job_id"])
        self.assertEqual(retried["attempts"], 2)
        self.assertEqual(retried["claim_generation"], job["claim_generation"] + 1)

    def test_lease_can_be_reclaimed_after_expiry(self) -> None:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.service.seal_batch("stat", "batch-a", 2)
        first = self.service.claim_job("worker-a", 10)
        self.clock.advance(seconds=11)
        second = self.service.claim_job("worker-b", 10)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(second["lease_owner"], "worker-b")
        self.assertEqual(second["claim_generation"], first["claim_generation"] + 1)
        # 旧持有人即使拿着旧的领取代次，也会被原子复核拒绝。
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-a", first["job_id"], "stat", first["claim_generation"])
        # 旧凭证同样不能把任务回报失败。
        with self.assertRaises(InvalidState):
            self.service.fail_job("worker-a", first["job_id"], "旧进程迟到的失败", first["claim_generation"])

    def _seal_with_job(self) -> dict:
        self.service.import_observations("operator", "batch-a", "key-1", self.rows)
        self.service.seal_batch("stat", "batch-a", 2)
        return self.service.get_batch("batch-a")

    def _count(self, sql: str, *params: object) -> int:
        return self.connection.execute(sql, params).fetchone()[0]

    def test_stale_worker_is_fenced_after_takeover(self) -> None:
        """危险窗口回归：A 租约内开算、写回前租约到期被 B 接管。

        使用冻结时钟推进，无需真实等待；接管顺序完全受控：A 领取 -> 时钟越过
        到期点 -> B 接管 -> A 用旧凭证迟到写回（且其结果已经算出）-> 被拒绝，
        B 凭新凭证完成，任务与分析记录最终只出现一次。
        """

        self._seal_with_job()

        # t0：A 领取租约（代次 g1）并开始计算。
        first = self.service.claim_job("worker-a", lease_seconds=10)
        self.assertEqual(first["claim_generation"], 1)

        # 计算耗时超过租约：仅推进时钟，不做任何 sleep。此时 A 已经算出结果但尚未写回。
        self.clock.advance(seconds=11)

        # B 发现租约过期并接管，拿到新的领取代次 g2。
        second = self.service.claim_job("worker-b", lease_seconds=10)
        self.assertEqual(second["claim_generation"], 2)
        self.assertEqual(second["lease_owner"], "worker-b")

        # A 持旧凭证迟到写回（complete_job 内部已先算完结果）：必须被拒绝。
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-a", first["job_id"], "stat", first["claim_generation"])

        # 拒绝后不得留下任何副作用：无分析记录、批次未被推进、无成功审计、任务仍归 B。
        self.assertEqual(self._count("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 0)
        self.assertEqual(self.service.get_batch("batch-a")["state"], "sealed")
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM audit_events WHERE entity_id='batch-a' "
                "AND event_type='analysis.completed'"
            ),
            0,
        )
        job_after = dict(
            self.connection.execute(
                "SELECT * FROM analysis_jobs WHERE job_id=?", (first["job_id"],)
            ).fetchone()
        )
        self.assertEqual(job_after["state"], "leased")
        self.assertEqual(job_after["lease_owner"], "worker-b")
        self.assertEqual(job_after["claim_generation"], 2)

        # 合法持有人 B 在租约内凭新代次完成。
        analysis = self.service.complete_job(
            "worker-b", second["job_id"], "stat", second["claim_generation"]
        )

        # 任务与分析记录只出现一次，批次只推进一次，成功审计只留下一条。
        self.assertEqual(self._count("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 1)
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM analysis_jobs WHERE batch_id='batch-a' AND state='succeeded'"
            ),
            1,
        )
        self.assertEqual(self.service.get_batch("batch-a")["state"], "analyzed")
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM audit_events WHERE entity_id='batch-a' "
                "AND event_type='analysis.completed'"
            ),
            1,
        )

        # 同一输入快照的算法是确定性的：B 的结果就是 A 当时本应写出的同一份结果，
        # 因此拒绝 A 不会丢失或改变统计结论。
        self.assertEqual(analysis["result"]["conclusion"], "pass")

        # B 完成之后，A 持旧代次再次迟到写回仍然被拒绝，且不会产生任何额外记录。
        with self.assertRaises(InvalidState):
            self.service.complete_job("worker-a", first["job_id"], "stat", first["claim_generation"])
        self.assertEqual(self._count("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 1)
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM audit_events WHERE entity_id='batch-a' "
                "AND event_type='analysis.completed'"
            ),
            1,
        )

    def test_stale_worker_is_fenced_during_compute_window(self) -> None:
        """精确还原危险窗口：A 通过租约检查后、在 analyze() 计算期间被 B 接管。

        旧实现在事务外完成全部租约校验，且写事务内忽略 UPDATE 的 rowcount，
        因此这种“算到一半被接管、随后迟到写回”会污染状态。新实现把裁决移进写
        事务并要求领取代次匹配，接管发生在计算中途也会被原子拒绝。
        """

        import robot_trials.service as service_mod

        self._seal_with_job()
        first = self.service.claim_job("worker-a", lease_seconds=10)
        self.assertEqual(first["claim_generation"], 1)

        original_analyze = service_mod.analyze
        taken_over: dict[str, object] = {}

        def take_over_during_compute(protocol, observations):  # type: ignore[no-untyped-def]
            # A 已通过事务外读取并进入计算：此刻租约到期，B 完成接管（不等待）。
            self.clock.advance(seconds=11)
            second = self.service.claim_job("worker-b", lease_seconds=10)
            assert second is not None and second["claim_generation"] == first["claim_generation"] + 1
            taken_over["job"] = second
            # A 仍按原输入快照把结果算完。
            return original_analyze(protocol, observations)

        service_mod.analyze = take_over_during_compute
        try:
            with self.assertRaises(InvalidState):
                self.service.complete_job(
                    "worker-a", first["job_id"], "stat", first["claim_generation"]
                )
        finally:
            service_mod.analyze = original_analyze

        # 拒绝必须是全有或全无的。
        self.assertEqual(self._count("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 0)
        self.assertEqual(self.service.get_batch("batch-a")["state"], "sealed")
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM audit_events WHERE entity_id='batch-a' "
                "AND event_type='analysis.completed'"
            ),
            0,
        )

        # 合法新持有人 B（接管时已领取）完成后，任务、分析与成功审计各只出现一次。
        second = taken_over["job"]
        self.service.complete_job("worker-b", second["job_id"], "stat", second["claim_generation"])
        self.assertEqual(self._count("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 1)
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM analysis_jobs WHERE batch_id='batch-a' AND state='succeeded'"
            ),
            1,
        )
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM audit_events WHERE entity_id='batch-a' "
                "AND event_type='analysis.completed'"
            ),
            1,
        )
        self.assertEqual(self.service.get_batch("batch-a")["state"], "analyzed")

    def test_legitimate_completion_is_idempotent_for_same_snapshot(self) -> None:
        """合法持有人用同一领取凭证重放完成时复用分析，不重复插入也不重复审计。"""

        self._seal_with_job()
        job = self.service.claim_job("worker-a", lease_seconds=30)

        first = self.service.complete_job("worker-a", job["job_id"], "stat", job["claim_generation"])
        # 客户端重试：持有人、代次不变，时钟仍在租约窗口之外也无妨（任务已成功）。
        second = self.service.complete_job("worker-a", job["job_id"], "stat", job["claim_generation"])

        self.assertEqual(second["analysis_id"], first["analysis_id"])
        self.assertEqual(second["input_sha256"], first["input_sha256"])
        self.assertEqual(second["result"], first["result"])
        self.assertEqual(self._count("SELECT count(*) FROM analyses WHERE batch_id='batch-a'"), 1)
        self.assertEqual(
            self._count(
                "SELECT count(*) FROM audit_events WHERE entity_id='batch-a' "
                "AND event_type='analysis.completed'"
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
