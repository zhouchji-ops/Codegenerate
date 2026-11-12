"""Environment-independent contract. Does NOT implement a physics verifier."""
from dataclasses import dataclass,field
@dataclass
class EpisodeContract:
    episode_id: str
    manifest_hash: str
    seed: int
    horizon: int=600
    steps: int=0
    ended: bool=False
    reward_paid: bool=False
    history: list=field(default_factory=list)
    action_queue: list=field(default_factory=list)
    def step(self, *, official_success, physical_verified, invalid_sim=False, executed_steps=1):
        if self.ended: raise RuntimeError("must reset after terminal/truncated episode")
        if not 1<=executed_steps<=self.horizon-self.steps: raise ValueError("invalid executed step count")
        self.steps+=executed_steps
        # Invalid simulation cannot earn success reward even if a predicate is spuriously true.
        valid_success=bool(official_success and physical_verified and not invalid_sim)
        reward=float(valid_success and not self.reward_paid)
        self.reward_paid|=bool(reward)
        terminated=bool(valid_success or invalid_sim)
        truncated=bool(self.steps>=self.horizon and not terminated)
        self.ended=terminated or truncated
        return {"reward":reward,"official_success":bool(official_success),
            "physical_verified":bool(physical_verified),"invalid_sim":bool(invalid_sim),
            "terminated":terminated,"truncated":truncated,"executed_steps":executed_steps,
            "episode_id":self.episode_id,"manifest_hash":self.manifest_hash,"seed":self.seed,
            "steps":self.steps}
    def reset(self, episode_id, manifest_hash, seed):
        self.episode_id=episode_id;self.manifest_hash=manifest_hash;self.seed=seed
        self.steps=0;self.ended=False;self.reward_paid=False
        self.history.clear();self.action_queue.clear()
        # Environment state and policy RNG must be explicitly reset by the caller.
