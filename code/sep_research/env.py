"""RLlib adapter for the Ray-free joint transition core."""

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from sep_research.env_core import SepResearchCore


class SepResearchEnv(SepResearchCore, MultiAgentEnv):
    """Classic MultiAgentEnv API; all episode behavior belongs to the core."""
