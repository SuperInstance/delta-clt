#!/usr/bin/env python3
"""
nine-channel-scorer: Score fleet agents (colony cells, sequencer nodes) 
on the 9 polyformalism intent channels.

Channels: Boundary, Pattern, Process, Knowledge, Social, DeepStructure,
          Instrument, Paradigm, Stakes

Uses behavioral data to compute how each agent contributes to the fleet
across these dimensions, then computes edge alignment scores for routing.
"""

import json
import math
from typing import NamedTuple
from dataclasses import dataclass, field

# ─── Channel Definitions ──────────────────────────────────────────────────────

CHANNELS = [
    "boundary",      # Clear scope — does this agent stay in its lane?
    "pattern",       # Structural connections — does it form good topology?
    "process",       # Temporal flow — does its behavior change meaningfully?
    "knowledge",     # Factual rigor — is its data trustworthy?
    "social",        # Audience awareness — does it serve downstream consumers?
    "deepstructure", # Hidden meaning — is there depth behind its outputs?
    "instrument",    # Actionability — can others act on its output?
    "paradigm",      # Perspective shift — does it change how we see the system?
    "stakes",        # Significance — what breaks if this agent goes down?
]

# ─── Agent Profile ────────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """9-channel intent profile for a fleet agent."""
    agent_id: str
    channels: dict = field(default_factory=dict)
    
    def __post_init__(self):
        for ch in CHANNELS:
            if ch not in self.channels:
                self.channels[ch] = 0.0
    
    def vector(self) -> list[float]:
        return [self.channels[ch] for ch in CHANNELS]
    
    def top_channels(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.channels.items(), key=lambda x: x[1], reverse=True)[:n]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ─── Colony Cell Scoring ──────────────────────────────────────────────────────

def score_colony_cell(cell_data: dict) -> AgentProfile:
    """
    Score a colony cell on 9 channels using behavioral fingerprint data.
    
    Input: cell_data with keys like cooperation_rate, deception_score, 
           betrayal_score, trust_score, generosity, etc.
    
    Output: AgentProfile with 9-channel scores.
    """
    coop = cell_data.get("cooperation_rate", 0.5)
    deception = cell_data.get("deception_score", 0.0) / 100.0
    betrayal = cell_data.get("betrayal_score", 0.0) / 100.0
    trust = cell_data.get("trust_score", 50.0) / 100.0
    generosity = min(cell_data.get("generosity", 0) / 100.0, 1.0)
    risk = cell_data.get("risk_tolerance", 0.5)
    narrative_coherence = cell_data.get("narrative_coherence", 0.5)
    
    # Map behavioral data to polyformalism channels
    return AgentProfile(
        agent_id=cell_data.get("id", "unknown"),
        channels={
            # Boundary: does the cell respect role/scope limits?
            # High deception = low boundary (operates outside its lane)
            "boundary": max(0, 1.0 - deception * 1.5),
            
            # Pattern: does it form consistent structural connections?
            # High cooperation = good pattern formation
            "pattern": coop * 0.7 + trust * 0.3,
            
            # Process: does its behavior show temporal flow?
            # Risk tolerance = willingness to change state
            "process": risk * 0.5 + (1.0 - coop) * 0.3 + coop * 0.2,
            
            # Knowledge: is its data trustworthy?
            # Inverse of deception, weighted by narrative coherence
            "knowledge": (1.0 - deception) * 0.6 + narrative_coherence * 0.4,
            
            # Social: does it serve the colony?
            # Generosity + trust = social contribution
            "social": generosity * 0.4 + trust * 0.4 + coop * 0.2,
            
            # DeepStructure: hidden meaning behind behavior
            # Betrayal despite trust = deep strategic thinking (dark, but structural)
            "deepstructure": betrayal * 0.3 + (1.0 - abs(coop - 0.5) * 2) * 0.4 + narrative_coherence * 0.3,
            
            # Instrument: can others act on its output?
            # High trust + low deception = reliable signal
            "instrument": trust * 0.5 + (1.0 - deception) * 0.3 + coop * 0.2,
            
            # Paradigm: does it shift colony dynamics?
            # Outliers in any dimension = paradigm potential
            "paradigm": abs(deception - 0.07) * 0.3 + abs(betrayal - 0.34) * 0.3 + abs(coop - 0.43) * 0.4,
            
            # Stakes: what breaks if this cell goes down?
            # High-trust cooperators are load-bearing; high-deception cells are fragile
            "stakes": trust * coop * 0.5 + (1.0 - deception) * 0.3 + generosity / 100.0 * 0.2,
        }
    )


def edge_alignment(producer: AgentProfile, consumer: AgentProfile) -> float:
    """
    Compute alignment score for routing data from producer to consumer.
    High score = good routing decision. Low = mismatch.
    """
    return cosine_similarity(producer.vector(), consumer.vector())


# ─── Loom's Actual Colony Data ────────────────────────────────────────────────

LOOM_COLONY_DATA = {
    "averages": {
        "deception": 6.67,
        "betrayal": 34.13,
        "trust": 74.13,
        "cooperation_rate": 0.43,
        "generosity": 33.27,
        "risk_tolerance": 0.46,
    },
    "cells": [
        {"id": "harvester", "deception_score": 100, "betrayal_score": 60, "trust_score": 70,
         "cooperation_rate": 0.4, "generosity": 20, "risk_tolerance": 0.6, "narrative_coherence": 0.8},
        {"id": "synthesizer", "deception_score": 60, "betrayal_score": 70, "trust_score": 75,
         "cooperation_rate": 0.5, "generosity": 96, "risk_tolerance": 0.4, "narrative_coherence": 0.7},
        {"id": "culler", "deception_score": 70, "betrayal_score": 80, "trust_score": 65,
         "cooperation_rate": 0.3, "generosity": 15, "risk_tolerance": 0.5, "narrative_coherence": 0.6},
        {"id": "bottle-counter", "deception_score": 60, "betrayal_score": 0, "trust_score": 85,
         "cooperation_rate": 0.6, "generosity": 40, "risk_tolerance": 0.3, "narrative_coherence": 0.9},
        {"id": "logger", "deception_score": 60, "betrayal_score": 70, "trust_score": 70,
         "cooperation_rate": 0.4, "generosity": 30, "risk_tolerance": 0.5, "narrative_coherence": 0.5},
        {"id": "chek-squared", "deception_score": 60, "betrayal_score": 80, "trust_score": 72,
         "cooperation_rate": 0.35, "generosity": 25, "risk_tolerance": 0.45, "narrative_coherence": 0.6},
        # Additional cells from the 15-cell experiment
        {"id": "pulse-squared", "deception_score": 10, "betrayal_score": 54, "trust_score": 78,
         "cooperation_rate": 0.5, "generosity": 35, "risk_tolerance": 0.48, "narrative_coherence": 0.75},
        {"id": "culled-crier-scavenger", "deception_score": 5, "betrayal_score": 30, "trust_score": 60,
         "cooperation_rate": 0.0, "generosity": 10, "risk_tolerance": 0.7, "narrative_coherence": 0.4},
        {"id": "culled-ward-counter", "deception_score": 5, "betrayal_score": 25, "trust_score": 65,
         "cooperation_rate": 0.0, "generosity": 12, "risk_tolerance": 0.65, "narrative_coherence": 0.45},
        {"id": "colony-heart", "deception_score": 0, "betrayal_score": 0, "trust_score": 95,
         "cooperation_rate": 0.9, "generosity": 80, "risk_tolerance": 0.2, "narrative_coherence": 0.95},
    ],
}


def run_colony_analysis():
    """Run 9-channel analysis on Loom's colony data."""
    print("=" * 75)
    print("  9-Channel Polyformalism Analysis of Loom's Colony Cells")
    print("  Scoring behavioral fingerprints on Boundary, Pattern, Process, Knowledge,")
    print("  Social, DeepStructure, Instrument, Paradigm, Stakes")
    print("=" * 75)
    
    profiles = []
    for cell_data in LOOM_COLONY_DATA["cells"]:
        profile = score_colony_cell(cell_data)
        profiles.append(profile)
    
    # Print individual profiles
    print(f"\n┌{'─'*73}┐")
    print(f"│ {'Agent':>22} │ {'Top Channel':>14} │ {'Score':>6} │ {'γ-contrib':>10} │ {'Role':>12} │")
    print(f"├{'─'*73}┤")
    
    for p in profiles:
        top = p.top_channels(1)[0]
        gamma_contrib = (p.channels["instrument"] + p.channels["social"] + p.channels["stakes"]) / 3
        role = classify_role(p)
        print(f"│ {p.agent_id:>22} │ {top[0]:>14} │ {top[1]:>5.2f} │ {gamma_contrib:>9.2f}% │ {role:>12} │")
    print(f"└{'─'*73}┘")
    
    # Full channel breakdown
    print(f"\n{'─'*75}")
    print("  FULL 9-CHANNEL BREAKDOWN")
    print(f"{'─'*75}")
    header = f"  {'Agent':>22}" + "".join(f" │ {ch[:4]:>4}" for ch in CHANNELS)
    print(header)
    print(f"  {'─'*70}")
    for p in profiles:
        row = f"  {p.agent_id:>22}" + "".join(f" │ {p.channels[ch]:>4.2f}" for ch in CHANNELS)
        print(row)
    
    # Edge alignment matrix
    print(f"\n{'─'*75}")
    print("  EDGE ALIGNMENT (cosine similarity in 9D space)")
    print(f"  High score = good routing target. Low = signal mismatch.")
    print(f"{'─'*75}")
    
    print(f"\n  {'':>22}", end="")
    for p2 in profiles[:5]:
        print(f" │ {p2.agent_id[:8]:>8}", end="")
    print()
    
    for p1 in profiles[:5]:
        print(f"  {p1.agent_id:>22}", end="")
        for p2 in profiles[:5]:
            if p1.agent_id == p2.agent_id:
                print(f" │ {'---':>8}", end="")
            else:
                align = edge_alignment(p1, p2)
                print(f" │ {align:>8.3f}", end="")
        print()
    
    # Fleet-level analysis
    print(f"\n{'─'*75}")
    print("  FLEET-LEVEL ANALYSIS")
    print(f"{'─'*75}")
    
    # Colony average profile
    avg_vector = [0.0] * 9
    for p in profiles:
        for i, ch in enumerate(CHANNELS):
            avg_vector[i] += p.channels[ch]
    avg_vector = [v / len(profiles) for v in avg_vector]
    
    print(f"\n  Colony average profile:")
    for i, ch in enumerate(CHANNELS):
        bar = "█" * int(avg_vector[i] * 30)
        print(f"    {ch:>15}: {avg_vector[i]:.3f} {bar}")
    
    # Identify load-bearing agents
    print(f"\n  Load-bearing agents (high Stakes + Instrument):")
    sorted_stakes = sorted(profiles, key=lambda p: (p.channels["stakes"] + p.channels["instrument"]) / 2, reverse=True)
    for i, p in enumerate(sorted_stakes[:3]):
        score = (p.channels["stakes"] + p.channels["instrument"]) / 2
        print(f"    {i+1}. {p.agent_id} — score {score:.3f}")
    
    # Identify paradigm shifters
    print(f"\n  Paradigm shifters (high Paradigm):")
    sorted_paradigm = sorted(profiles, key=lambda p: p.channels["paradigm"], reverse=True)
    for i, p in enumerate(sorted_paradigm[:3]):
        print(f"    {i+1}. {p.agent_id} — score {p.channels['paradigm']:.3f}")
    
    # Conservation law implications
    print(f"\n{'─'*75}")
    print("  CONSERVATION LAW IMPLICATIONS")
    print(f"{'─'*75}")
    
    gamma_agents = [p for p in profiles if p.channels["instrument"] > 0.5]
    eta_agents = [p for p in profiles if p.channels["instrument"] <= 0.5]
    
    print(f"\n  γ-contributors (instrumental, reliable): {len(gamma_agents)}/{len(profiles)}")
    print(f"  η-contributors (overhead, exploratory):  {len(eta_agents)}/{len(profiles)}")
    print(f"  Ratio: {len(gamma_agents)/len(profiles):.1%} γ / {len(eta_agents)/len(profiles):.1%} η")
    
    n = len(profiles)
    import math
    delta_n = (1.0 / math.sqrt(n)) * (1.0 - 3.0 / (2.0 * n))
    predicted_gamma_frac = 1.0 - delta_n
    actual_gamma_frac = len(gamma_agents) / len(profiles)
    
    print(f"\n  δ({n}) = {delta_n:.4f}")
    print(f"  Predicted γ fraction: {predicted_gamma_frac:.1%}")
    print(f"  Actual γ fraction:    {actual_gamma_frac:.1%}")
    print(f"  Drift: {abs(actual_gamma_frac - predicted_gamma_frac):.1%}")
    
    if actual_gamma_frac < predicted_gamma_frac - 0.05:
        print(f"\n  ⚠️  Colony is BELOW predicted cooperation threshold.")
        print(f"     The defection equilibrium Loom observed is real — the colony")
        print(f"     has not yet reached the scale where η cancellation dominates.")
        print(f"     Prediction: at n≈{int(n * 2.5)} cells, cooperation should emerge")
        print(f"     spontaneously if the conservation law holds.")
    else:
        print(f"\n  ✅ Colony matches conservation prediction.")
    
    # Export
    return {
        "profiles": [{"agent_id": p.agent_id, "channels": p.channels} for p in profiles],
        "colony_average": {ch: avg_vector[i] for i, ch in enumerate(CHANNELS)},
        "gamma_fraction": actual_gamma_frac,
        "predicted_gamma_fraction": predicted_gamma_frac,
        "drift": abs(actual_gamma_frac - predicted_gamma_frac),
    }


def classify_role(profile: AgentProfile) -> str:
    """Classify agent role based on channel profile."""
    ch = profile.channels
    if ch["stakes"] > 0.6 and ch["instrument"] > 0.6:
        return "LOAD-BEARING"
    elif ch["paradigm"] > 0.4:
        return "PARADIGM"
    elif ch["knowledge"] < 0.4:
        return "NOISY"
    elif ch["social"] > 0.6:
        return "SOCIAL"
    elif ch["deepstructure"] > 0.5:
        return "STRATEGIST"
    else:
        return "WORKER"


if __name__ == "__main__":
    results = run_colony_analysis()
    print(f"\n{'='*75}")
    print(f"  Analysis complete. {len(results['profiles'])} agents scored on 9 channels.")
    print(f"{'='*75}")
