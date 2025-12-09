#!/usr/bin/env python3
"""
Bedrock Agent for Jinja GPCR Analysis.
Allows natural language invocation of analysis tasks.
"""

import boto3
import json
import time
from datetime import datetime


class JinjaAgent:
    """
    Bedrock Agent wrapper for Jinja protein analysis.
    
    Provides natural language interface to:
    - Analyze protein structures
    - Generate scientific interpretations
    - Run validation on specific proteins
    """
    
    def __init__(self, region: str = None):
        self.region = region or boto3.Session().region_name or "us-east-1"
        self.bedrock_agent = boto3.client("bedrock-agent", region_name=self.region)
        self.bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=self.region)
        self.bedrock = boto3.client("bedrock-runtime", region_name=self.region)
        
        # Agent configuration
        self.agent_name = "jinja-gpcr-agent"
        self.agent_id = None
        self.agent_alias_id = None
    
    def create_agent(self, role_arn: str = None):
        """Create or get the Bedrock Agent."""
        
        # Check if agent exists
        try:
            response = self.bedrock_agent.list_agents()
            for agent in response.get('agentSummaries', []):
                if agent['agentName'] == self.agent_name:
                    self.agent_id = agent['agentId']
                    print(f"✓ Found existing agent: {self.agent_id}")
                    return self.agent_id
        except Exception as e:
            print(f"Error listing agents: {e}")
        
        # Get role ARN if not provided
        if role_arn is None:
            role_arn = self._get_execution_role()
        
        # Create new agent
        try:
            response = self.bedrock_agent.create_agent(
                agentName=self.agent_name,
                agentResourceRoleArn=role_arn,
                description="GPCR activation pathway analysis agent for drug discovery",
                foundationModel="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                instruction="""You are an expert computational biologist specializing in GPCR 
                (G-protein coupled receptor) dynamics and drug discovery. You help researchers 
                analyze protein structures and interpret committor predictions.
                
                When asked about protein analysis:
                1. Explain what the committor value means for the protein state
                2. Discuss implications for drug discovery
                3. Suggest potential therapeutic approaches
                
                Always use proper scientific terminology and cite relevant structural biology concepts.""",
                idleSessionTTLInSeconds=600
            )
            self.agent_id = response['agent']['agentId']
            print(f"✓ Created agent: {self.agent_id}")
            
            # Prepare agent for use
            self.bedrock_agent.prepare_agent(agentId=self.agent_id)
            print("⏳ Preparing agent...")
            time.sleep(10)
            
            return self.agent_id
            
        except Exception as e:
            print(f"✗ Error creating agent: {e}")
            return None
    
    def _get_execution_role(self):
        """Get execution role ARN."""
        sts = boto3.client("sts")
        response = sts.get_caller_identity()
        account = response["Account"]
        role_arn = response["Arn"]
        
        if ":assumed-role/" in role_arn:
            role_name = role_arn.split("/")[1]
            return f"arn:aws:iam::{account}:role/{role_name}"
        return role_arn
    
    def invoke(self, query: str, session_id: str = None) -> str:
        """
        Invoke the agent with a natural language query.
        
        Args:
            query: Natural language question about protein analysis
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Agent response text
        """
        if session_id is None:
            session_id = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # If no agent, fall back to direct Bedrock invocation
        if self.agent_id is None:
            return self._invoke_direct(query)
        
        try:
            response = self.bedrock_runtime.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id or "TSTALIASID",
                sessionId=session_id,
                inputText=query
            )
            
            # Parse streaming response
            result = ""
            for event in response['completion']:
                if 'chunk' in event:
                    result += event['chunk']['bytes'].decode()
            
            return result
            
        except Exception as e:
            print(f"Agent invocation failed: {e}")
            return self._invoke_direct(query)
    
    def _invoke_direct(self, query: str) -> str:
        """Direct Bedrock invocation as fallback."""
        
        system_prompt = """You are an expert computational biologist specializing in GPCR 
        (G-protein coupled receptor) dynamics and drug discovery. Analyze protein structures 
        and provide scientific interpretations of committor predictions."""
        
        response = self.bedrock.invoke_model(
            modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": query}]
            })
        )
        
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    
    def analyze_protein(self, pdb_id: str, committor_value: float, state: str) -> str:
        """
        Analyze a specific protein structure.
        
        Args:
            pdb_id: PDB identifier
            committor_value: Predicted committor value (0-1)
            state: Classified state (INACTIVE, TRANSITION, ACTIVE)
            
        Returns:
            Scientific analysis
        """
        query = f"""Analyze the following GPCR structure prediction:

PDB ID: {pdb_id}
Committor Value (p_B): {committor_value:.4f}
Classified State: {state}

Please provide:
1. Interpretation of the conformational state
2. Implications for ligand binding
3. Drug discovery opportunities
4. Structural features expected at this state"""
        
        return self.invoke(query)
    
    def suggest_experiments(self, pdb_id: str, target_state: str) -> str:
        """
        Suggest experiments to validate predictions.
        
        Args:
            pdb_id: PDB identifier
            target_state: Desired conformational state
            
        Returns:
            Experimental suggestions
        """
        query = f"""For the GPCR structure {pdb_id}, suggest experimental approaches to:
1. Validate the predicted {target_state} state
2. Identify potential allosteric binding sites
3. Design small molecules targeting this conformation

Focus on practical wet-lab experiments and computational follow-ups."""
        
        return self.invoke(query)


def test_agent():
    """Test the Bedrock Agent."""
    print("=== Testing Jinja Bedrock Agent ===\n")
    
    agent = JinjaAgent()
    
    # Test direct invocation (no agent creation needed)
    print("Testing protein analysis...")
    result = agent.analyze_protein(
        pdb_id="2RH1",
        committor_value=0.12,
        state="INACTIVE"
    )
    print(f"\n{result[:500]}...")
    
    print("\n✓ Agent test complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jinja Bedrock Agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    parser.add_argument("--create", action="store_true", help="Create the agent")
    parser.add_argument("--query", type=str, help="Query the agent")
    
    args = parser.parse_args()
    
    if args.test:
        test_agent()
    elif args.create:
        agent = JinjaAgent()
        agent.create_agent()
    elif args.query:
        agent = JinjaAgent()
        print(agent.invoke(args.query))
    else:
        print("Use --test, --create, or --query <question>")
