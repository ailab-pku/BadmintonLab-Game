"""Per-court rally scores shared by playback and video recording."""

import torch


class MatchScore:
    def __init__(self, num_envs, device, num_rallies):
        self.scores = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.num_rallies = num_rallies

    @property
    def finished(self):
        return self.scores.sum(dim=1) >= self.num_rallies

    @property
    def all_finished(self):
        return bool(self.finished.all())

    def update(self, info):
        env_ids = (info["rally_resolved"] & ~self.finished).nonzero(as_tuple=False).flatten()
        winners = info["rally_winner"][env_ids]
        self.scores[env_ids, winners] += 1
        for env_id, winner, score in zip(env_ids.tolist(), winners.tolist(), self.scores[env_ids].tolist()):
            print(
                f"[SCORE] Court {env_id}, rally {sum(score)}/{self.num_rallies}: "
                f"player_{winner} wins; player_0 {score[0]} : {score[1]} player_1",
                flush=True,
            )
            if sum(score) == self.num_rallies:
                self._result(env_id, *score)

    @staticmethod
    def _result(env_id, score0, score1):
        result = "draw" if score0 == score1 else f"player_{int(score1 > score0)} wins"
        print(f"[RESULT] Court {env_id}: {result} ({score0}:{score1})", flush=True)

    def summary(self):
        for env_id, (score0, score1) in enumerate(self.scores.tolist()):
            if score0 + score1 >= self.num_rallies:
                self._result(env_id, score0, score1)
            else:
                print(
                    f"[SCORE] Court {env_id}: incomplete ({score0 + score1}/{self.num_rallies} rallies); "
                    f"player_0 {score0} : {score1} player_1",
                    flush=True,
                )
