#!/usr/bin/env python3
"""
Bedrock client utility for generating scientific interpretations using Claude.
"""

import boto3
import json
from typing import Optional


def get_bedrock_client(region: str = None):
    """Get Bedrock runtime client."""
    if region is None:
        region = boto3.Session().region_name or "us-east-1"
    return boto3.client("bedrock-runtime", region_name=region)


def generate_interpretation(
    protein_name: str,
    pdb_id: str,
    committor_value: float,
    state: str,
    description: str = "",
    model_id: str = "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    region: str = None
) -> str:
    """
    Generate a scientific interpretation of a committor prediction using Claude.
    
    Args:
        protein_name: Name of the protein (e.g., "β2-AR")
        pdb_id: PDB ID of the structure
        committor_value: Predicted committor value (0-1)
        state: Classified state (INACTIVE, TRANSITION, ACTIVE)
        description: Additional description of the structure
        model_id: Bedrock model ID
        region: AWS region
    
    Returns:
        Generated scientific interpretation
    """
    client = get_bedrock_client(region)
    
    prompt = f"""You are an expert computational biologist specializing in GPCR (G-protein coupled receptor) dynamics and drug discovery.

Analyze the following committor prediction result for a protein structure:

**Protein**: {protein_name}
**PDB ID**: {pdb_id}
**Description**: {description}
**Committor Value (p_B)**: {committor_value:.4f}
**Classified State**: {state}

The committor value represents the probability of transitioning from the inactive (A) to active (B) state:
- p_B ≈ 0: Strongly inactive conformation
- p_B ≈ 0.5: Transition state (at the free energy barrier)
- p_B ≈ 1: Strongly active conformation

Provide a concise scientific interpretation (3-4 bullet points) covering:
1. What this committor value tells us about the receptor's conformational state
2. Implications for ligand binding and G-protein coupling
3. Drug discovery relevance (what types of drugs might target this state)
4. Any notable structural features expected at this committor value

Be specific to GPCRs and use proper scientific terminology. Keep each bullet point to 1-2 sentences."""

    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
        
    except Exception as e:
        # Fallback to static interpretation if Bedrock fails
        return generate_fallback_interpretation(state, committor_value)


def generate_fallback_interpretation(state: str, committor_value: float) -> str:
    """Generate static fallback interpretation."""
    if state == "INACTIVE":
        return f"""• p_B={committor_value:.2f} indicates the receptor is in a resting/inactive conformation
• Orthosteric site likely occupied by antagonist or inverse agonist
• G-protein binding site (intracellular) is occluded
• Target for inverse agonists or antagonist development"""
    elif state == "ACTIVE":
        return f"""• p_B={committor_value:.2f} indicates the receptor is in an active/signaling conformation
• Agonist-bound state with open intracellular domain for G-protein coupling
• TM6 outward movement allows Gα binding
• Target for agonist optimization or biased agonist development"""
    else:  # TRANSITION
        return f"""• p_B={committor_value:.2f} places the receptor at the transition state barrier
• Allosteric binding sites are maximally exposed at this state
• Ideal target for allosteric modulators (PAMs or NAMs)
• Rate-determining step for receptor activation"""


def test_bedrock_access(region: str = None):
    """Test Bedrock access with a simple prompt."""
    print("=== Testing Bedrock Access ===")
    
    try:
        client = get_bedrock_client(region)
        
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Say 'Bedrock connection successful' and nothing else."}
                ]
            })
        )
        
        result = json.loads(response["body"].read())
        print(f"✓ {result['content'][0]['text']}")
        return True
        
    except Exception as e:
        print(f"✗ Bedrock access failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bedrock client utilities")
    parser.add_argument("--test", action="store_true", help="Test Bedrock access")
    parser.add_argument("--region", default=None, help="AWS region")
    
    args = parser.parse_args()
    
    if args.test:
        test_bedrock_access(args.region)
