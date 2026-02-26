"""Universal tools that work with any LLM, even those without native tool support."""

import re
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class ThinkingResult:
    """Result of thinking process."""
    thought_process: str
    final_answer: str
    confidence: float
    steps: List[str]


class ThinkingTool:
    """Force any LLM to think before answering."""
    
    THINKING_PROMPT = """Before answering, please think through this step by step.

Problem: {question}

Please analyze this by:
1. Understanding what is being asked
2. Identifying key information
3. Breaking down the problem
4. Considering different angles
5. Formulating your answer

Provide your thinking process inside <thinking> tags, then your final answer inside <answer> tags.

Format:
<thinking>
[Your detailed thinking process here]
</thinking>

<answer>
[Your concise final answer here]
</answer>"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def think(self, question: str) -> ThinkingResult:
        """Make the LLM think through a problem."""
        prompt = self.THINKING_PROMPT.format(question=question)
        
        response = self.llm.chat(prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse the response
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        
        thought_process = thinking_match.group(1).strip() if thinking_match else "No thinking process found"
        final_answer = answer_match.group(1).strip() if answer_match else text
        
        # Extract steps
        steps = self._extract_steps(thought_process)
        
        # Calculate confidence based on detail level
        confidence = min(1.0, len(thought_process) / 500)
        
        return ThinkingResult(
            thought_process=thought_process,
            final_answer=final_answer,
            confidence=confidence,
            steps=steps,
        )
    
    def _extract_steps(self, thinking: str) -> List[str]:
        """Extract reasoning steps from thinking."""
        steps = []
        
        # Look for numbered lists
        numbered = re.findall(r'\d+\.\s*(.+)', thinking)
        if numbered:
            steps.extend(numbered)
        
        # Look for bullet points
        bullets = re.findall(r'[\-\*]\s*(.+)', thinking)
        if bullets:
            steps.extend(bullets)
        
        # If no structure found, split by newlines
        if not steps:
            steps = [s.strip() for s in thinking.split('\n') if s.strip()]
        
        return steps[:10]  # Limit to 10 steps
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for agent."""
        return {
            "name": "think",
            "description": "Think through a problem step by step before answering",
            "parameters": {
                "question": {
                    "type": "string",
                    "description": "The question or problem to think through"
                }
            },
            "required": ["question"]
        }


class VerifiedSearchTool:
    """Web search with verification from trusted sources."""
    
    VERIFIED_SOURCES = {
        "wikipedia": {
            "domain": "wikipedia.org",
            "trust_score": 0.9,
        },
        "arxiv": {
            "domain": "arxiv.org",
            "trust_score": 0.95,
        },
        "google_scholar": {
            "domain": "scholar.google.com",
            "trust_score": 0.9,
        },
        "government": {
            "domains": [".gov", ".gov.uk", ".gob.mx"],
            "trust_score": 0.95,
        },
        "education": {
            "domains": [".edu", ".ac.uk", ".ac.jp"],
            "trust_score": 0.85,
        },
        "news": {
            "domains": ["reuters.com", "apnews.com", "bbc.com", "cnn.com"],
            "trust_score": 0.75,
        },
    }
    
    def __init__(self, search_tool: Optional[Any] = None):
        self.search_tool = search_tool
    
    def search_verified(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        min_trust_score: float = 0.7,
    ) -> Dict[str, Any]:
        """Search only verified sources."""
        # Determine which sources to use
        target_sources = sources or list(self.VERIFIED_SOURCES.keys())
        
        # Modify query to target specific sites
        site_restrictions = []
        for source in target_sources:
            if source in self.VERIFIED_SOURCES:
                info = self.VERIFIED_SOURCES[source]
                if "domain" in info:
                    site_restrictions.append(f"site:{info['domain']}")
                elif "domains" in info:
                    for domain in info["domains"]:
                        site_restrictions.append(f"site:{domain}")
        
        # Build search query
        if site_restrictions:
            search_query = f"{query} ({' OR '.join(site_restrictions)})"
        else:
            search_query = query
        
        # Perform search
        if self.search_tool:
            results = self.search_tool.search(search_query, num_results=10)
        else:
            # Use default web search
            from ..tools.web_search import WebSearch
            searcher = WebSearch()
            results = searcher.search(search_query, num_results=10)
        
        # Verify and score results
        verified_results = []
        for result in results:
            trust_info = self._calculate_trust_score(result.url)
            
            if trust_info["score"] >= min_trust_score:
                verified_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "trust_score": trust_info["score"],
                    "source_type": trust_info["type"],
                })
        
        # Sort by trust score
        verified_results.sort(key=lambda x: x["trust_score"], reverse=True)
        
        return {
            "query": query,
            "verified_results": verified_results,
            "unverified_count": len(results) - len(verified_results),
            "confidence": self._calculate_overall_confidence(verified_results),
        }
    
    def _calculate_trust_score(self, url: str) -> Dict[str, Any]:
        """Calculate trust score for a URL."""
        url_lower = url.lower()
        
        for source_type, info in self.VERIFIED_SOURCES.items():
            if "domain" in info:
                if info["domain"] in url_lower:
                    return {"score": info["trust_score"], "type": source_type}
            elif "domains" in info:
                for domain in info["domains"]:
                    if domain in url_lower:
                        return {"score": info["trust_score"], "type": source_type}
        
        # Default low score for unknown sources
        return {"score": 0.3, "type": "unknown"}
    
    def _calculate_overall_confidence(self, results: List[Dict]) -> float:
        """Calculate overall confidence from results."""
        if not results:
            return 0.0
        
        # Average trust scores
        avg_score = sum(r["trust_score"] for r in results) / len(results)
        
        # Boost for multiple high-quality sources
        high_quality = sum(1 for r in results if r["trust_score"] > 0.8)
        boost = min(0.1, high_quality * 0.02)
        
        return min(1.0, avg_score + boost)
    
    def verify_fact(self, claim: str) -> Dict[str, Any]:
        """Verify a specific claim against verified sources."""
        # Search for the claim
        search_result = self.search_verified(claim, min_trust_score=0.8)
        
        # Extract supporting and contradicting evidence
        supporting = []
        contradicting = []
        
        for result in search_result["verified_results"]:
            # Simple heuristic: if claim keywords appear in result
            if self._contains_claim_evidence(result["snippet"], claim):
                supporting.append(result)
        
        # Determine verdict
        if len(supporting) >= 2:
            verdict = "verified"
        elif len(supporting) == 1:
            verdict = "likely_true"
        elif search_result["verified_results"]:
            verdict = "unverified"
        else:
            verdict = "insufficient_evidence"
        
        return {
            "claim": claim,
            "verdict": verdict,
            "supporting_evidence": supporting,
            "confidence": search_result["confidence"],
            "sources_checked": len(search_result["verified_results"]),
        }
    
    def _contains_claim_evidence(self, text: str, claim: str) -> bool:
        """Check if text contains evidence for a claim."""
        # Extract key terms from claim
        key_terms = [w.lower() for w in claim.split() if len(w) > 3]
        
        # Check if key terms appear in text
        text_lower = text.lower()
        matches = sum(1 for term in key_terms if term in text_lower)
        
        # If more than half the key terms match, consider it evidence
        return matches >= len(key_terms) * 0.5
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for agent."""
        return {
            "name": "verified_search",
            "description": "Search verified sources like Wikipedia, academic papers, and government sites",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["wikipedia", "arxiv", "google_scholar", "government", "education"]
                    },
                    "description": "Specific sources to search (optional)"
                }
            },
            "required": ["query"]
        }


class ChainOfThoughtTool:
    """Chain of Thought prompting for any LLM."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def solve(
        self,
        problem: str,
        steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Solve problem using chain of thought."""
        if steps is None:
            steps = self._generate_steps(problem)
        
        # Build chain of thought prompt
        cot_prompt = f"Problem: {problem}\n\nLet's solve this step by step:\n\n"
        
        for i, step in enumerate(steps, 1):
            cot_prompt += f"Step {i}: {step}\n"
        
        cot_prompt += "\nNow, based on these steps, what is the answer?"
        
        # Get response
        response = self.llm.chat(cot_prompt)
        answer = response.text if hasattr(response, 'text') else str(response)
        
        return {
            "problem": problem,
            "steps": steps,
            "answer": answer,
            "full_reasoning": cot_prompt + "\n" + answer,
        }
    
    def _generate_steps(self, problem: str) -> List[str]:
        """Generate reasoning steps."""
        # Use LLM to generate steps
        step_prompt = f"Break down this problem into 3-5 clear steps: {problem}\n\nSteps:"
        
        response = self.llm.chat(step_prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse steps
        lines = text.split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered items
            if re.match(r'^\d+[\.\)]\s+', line):
                step = re.sub(r'^\d+[\.\)]\s+', '', line)
                steps.append(step)
        
        return steps if steps else ["Analyze the problem", "Formulate solution", "Verify answer"]


# Convenience functions
def think_before_answer(llm, question: str) -> ThinkingResult:
    """Quick thinking tool usage."""
    tool = ThinkingTool(llm)
    return tool.think(question)


def search_verified(query: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """Quick verified search."""
    tool = VerifiedSearchTool()
    return tool.search_verified(query, sources)


def chain_of_thought(llm, problem: str) -> Dict[str, Any]:
    """Quick chain of thought."""
    tool = ChainOfThoughtTool(llm)
    return tool.solve(problem)
