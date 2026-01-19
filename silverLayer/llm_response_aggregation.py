"""
Multi-LLM Response Aggregation and Judgment
============================================

This module demonstrates how to:
1. Get responses from two different LLMs
2. Judge/evaluate the responses
3. Aggregate them into a final response using different methods

Three Aggregation Methods:
1. Selection - Choose the best response based on judgment
2. Synthesis - Merge both responses into one comprehensive answer
3. Weighted Combination - Combine responses using weighted scores

Author: LangChain Reference
Date: 2024
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()


# ============================================================================
# 1. INITIALIZE MULTIPLE LLMs
# ============================================================================

def initialize_llms():
    """
    Initialize different LLMs for comparison
    
    Returns:
        tuple: (google_llm, openai_llm, groq_llm)
    """
    
    # Google Generative AI
    google_llm = ChatGoogleGenerativeAI(
        model="models/gemini-pro",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # OpenAI (optional)
    openai_llm = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI: {e}")
    
    # Groq (optional)
    groq_llm = None
    if os.getenv("GROQ_API_KEY"):
        try:
            groq_llm = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.7,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            print(f"Warning: Could not initialize Groq: {e}")
    
    return google_llm, openai_llm, groq_llm


# ============================================================================
# 2. GET RESPONSES FROM MULTIPLE LLMs
# ============================================================================

def get_dual_responses(question, llm1, llm2, llm1_name="LLM 1", llm2_name="LLM 2"):
    """
    Get responses from two LLMs
    
    Args:
        question (str): The question to ask both LLMs
        llm1: First language model
        llm2: Second language model
        llm1_name (str): Name of first LLM for display
        llm2_name (str): Name of second LLM for display
    
    Returns:
        tuple: (response1, response2)
    """
    
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}")
    
    # Get response from first LLM
    try:
        response1 = llm1.invoke(question)
        answer1 = response1.content
        print(f"\n✓ Response from {llm1_name}:\n{answer1}")
    except Exception as e:
        print(f"✗ Error from {llm1_name}: {e}")
        answer1 = ""
    
    # Get response from second LLM
    try:
        response2 = llm2.invoke(question)
        answer2 = response2.content
        print(f"\n✓ Response from {llm2_name}:\n{answer2}")
    except Exception as e:
        print(f"✗ Error from {llm2_name}: {e}")
        answer2 = ""
    
    return answer1, answer2


# ============================================================================
# 3. JUDGE/EVALUATE RESPONSES
# ============================================================================

def judge_responses(question, response1, response2, judge_llm):
    """
    Use an LLM to evaluate and compare two responses
    
    Args:
        question (str): Original question
        response1 (str): First response
        response2 (str): Second response
        judge_llm: LLM to use for judging
    
    Returns:
        str: JSON formatted judgment
    """
    
    judge_prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator. Compare the following two responses to a question.
    
    Question: {question}
    
    Response 1:
    {response1}
    
    Response 2:
    {response2}
    
    Please evaluate both responses on:
    1. Accuracy and correctness
    2. Completeness and coverage
    3. Clarity and organization
    4. Helpfulness
    
    Provide scores (1-10) for each criterion and explain your judgment.
    
    Format your response as JSON:
    {{
        "response1_score": <number>,
        "response2_score": <number>,
        "response1_strengths": "...",
        "response1_weaknesses": "...",
        "response2_strengths": "...",
        "response2_weaknesses": "...",
        "better_response": "response1" or "response2",
        "reason_for_choice": "..."
    }}
    """)
    
    chain = judge_prompt | judge_llm | StrOutputParser()
    
    print("\n" + "="*80)
    print("JUDGMENT PHASE")
    print("="*80)
    
    judgment = chain.invoke({
        "question": question,
        "response1": response1,
        "response2": response2
    })
    
    print(f"\nJudgment:\n{judgment}")
    
    return judgment


# ============================================================================
# 4. AGGREGATE RESPONSES - METHOD 1: SELECT BEST
# ============================================================================

def aggregate_by_selection(question, response1, response2, judgment, judge_llm):
    """
    Aggregate by selecting the better response based on judgment
    
    Best for: When one response is clearly superior
    
    Args:
        question (str): Original question
        response1 (str): First response
        response2 (str): Second response
        judgment (str): JSON judgment from judge_llm
        judge_llm: The judging LLM
    
    Returns:
        str: Selected final response
    """
    
    print("\n" + "="*80)
    print("AGGREGATION METHOD 1: SELECTION")
    print("(Choose the best response)")
    print("="*80)
    
    # Extract the better response from judgment
    try:
        judgment_json = json.loads(judgment)
        better = judgment_json.get("better_response", "response1")
        reason = judgment_json.get("reason_for_choice", "")
        
        if better == "response1":
            final_response = response1
            print(f"\n✓ Selected Response 1")
        else:
            final_response = response2
            print(f"\n✓ Selected Response 2")
        
        print(f"Reason: {reason}")
    except Exception as e:
        print(f"Warning: Could not parse judgment JSON: {e}")
        final_response = response1 if "response1" in judgment else response2
    
    print(f"\nFinal Response (Selected):\n{final_response}")
    return final_response


# ============================================================================
# 5. AGGREGATE RESPONSES - METHOD 2: SYNTHESIS/MERGE
# ============================================================================

def aggregate_by_synthesis(question, response1, response2, judge_llm):
    """
    Aggregate by synthesizing/merging both responses
    
    Best for: When both responses have valuable information
    
    Args:
        question (str): Original question
        response1 (str): First response
        response2 (str): Second response
        judge_llm: The synthesizing LLM
    
    Returns:
        str: Synthesized final response
    """
    
    print("\n" + "="*80)
    print("AGGREGATION METHOD 2: SYNTHESIS")
    print("(Merge both responses into one comprehensive answer)")
    print("="*80)
    
    synthesis_prompt = ChatPromptTemplate.from_template("""
    You are a synthesis expert. Combine the best aspects of these two responses
    into one comprehensive, high-quality answer.
    
    Question: {question}
    
    Response 1:
    {response1}
    
    Response 2:
    {response2}
    
    Instructions:
    1. Identify key points from both responses
    2. Remove redundancies and duplications
    3. Combine complementary information
    4. Fill gaps using the best from each response
    5. Ensure logical flow and clarity
    6. Maintain a coherent structure
    
    Provide a single synthesized response that is better and more complete than both originals.
    """)
    
    chain = synthesis_prompt | judge_llm | StrOutputParser()
    
    synthesized = chain.invoke({
        "question": question,
        "response1": response1,
        "response2": response2
    })
    
    print(f"\nFinal Response (Synthesized):\n{synthesized}")
    return synthesized


# ============================================================================
# 6. AGGREGATE RESPONSES - METHOD 3: WEIGHTED COMBINATION
# ============================================================================

def aggregate_by_weighted_combination(question, response1, response2, judgment, judge_llm):
    """
    Aggregate using weighted scoring from judgment
    
    Best for: When you want to incorporate strong points from both responses
    
    Args:
        question (str): Original question
        response1 (str): First response
        response2 (str): Second response
        judgment (str): JSON judgment with scores
        judge_llm: The combining LLM
    
    Returns:
        str: Weighted combined response
    """
    
    print("\n" + "="*80)
    print("AGGREGATION METHOD 3: WEIGHTED COMBINATION")
    print("(Combine using weighted scores)")
    print("="*80)
    
    try:
        judgment_json = json.loads(judgment)
        score1 = judgment_json.get("response1_score", 5)
        score2 = judgment_json.get("response2_score", 5)
        
        # Normalize scores to weights
        total = score1 + score2
        weight1 = score1 / total if total > 0 else 0.5
        weight2 = score2 / total if total > 0 else 0.5
        
        print(f"\nResponse 1 Score: {score1}/10 (Weight: {weight1:.1%})")
        print(f"Response 2 Score: {score2}/10 (Weight: {weight2:.1%})")
        
    except Exception as e:
        print(f"Warning: Could not extract scores: {e}")
        weight1, weight2 = 0.5, 0.5
    
    # Create weighted combination prompt
    weighted_prompt = ChatPromptTemplate.from_template("""
    Combine these two responses with the following relative importance:
    - Response 1 (Weight: {weight1:.0%}): 
    {response1}
    
    - Response 2 (Weight: {weight2:.0%}): 
    {response2}
    
    Question: {question}
    
    Create a final answer that:
    1. Emphasizes points from Response 1 ({weight1:.0%} importance)
    2. Includes points from Response 2 ({weight2:.0%} importance)
    3. Maintains natural flow and coherence
    4. Avoids redundancy
    
    The final answer should reflect the weighted importance of each response.
    """)
    
    chain = weighted_prompt | judge_llm | StrOutputParser()
    
    weighted_response = chain.invoke({
        "question": question,
        "response1": response1,
        "response2": response2,
        "weight1": weight1,
        "weight2": weight2
    })
    
    print(f"\nFinal Response (Weighted Combination):\n{weighted_response}")
    return weighted_response


# ============================================================================
# 7. COMPLETE WORKFLOW
# ============================================================================

def complete_aggregation_workflow(question, llm1=None, llm2=None, judge_llm=None):
    """
    Execute complete workflow: dual responses → judgment → aggregation
    
    Args:
        question (str): Question to ask
        llm1: First LLM (default: Google)
        llm2: Second LLM (default: Groq or Google)
        judge_llm: LLM for judging (default: Google)
    
    Returns:
        dict: Dictionary containing all responses and aggregations
    """
    
    print("\n" + "="*80)
    print("COMPLETE MULTI-LLM AGGREGATION WORKFLOW")
    print("="*80)
    
    # Initialize LLMs if not provided
    if llm1 is None or llm2 is None or judge_llm is None:
        google_llm, openai_llm, groq_llm = initialize_llms()
        llm1 = llm1 or google_llm
        llm2 = llm2 or groq_llm or google_llm
        judge_llm = judge_llm or google_llm
    
    # Step 1: Get dual responses
    response1, response2 = get_dual_responses(question, llm1, llm2)
    
    if not response1 or not response2:
        print("✗ Could not get responses from both LLMs")
        return None
    
    # Step 2: Judge responses
    judgment = judge_responses(question, response1, response2, judge_llm)
    
    # Step 3: Aggregate using different methods
    
    # Method 1: Selection
    final_by_selection = aggregate_by_selection(
        question, response1, response2, judgment, judge_llm
    )
    
    # Method 2: Synthesis
    final_by_synthesis = aggregate_by_synthesis(
        question, response1, response2, judge_llm
    )
    
    # Method 3: Weighted
    final_by_weighted = aggregate_by_weighted_combination(
        question, response1, response2, judgment, judge_llm
    )
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    
    return {
        "question": question,
        "response1": response1,
        "response2": response2,
        "judgment": judgment,
        "final_by_selection": final_by_selection,
        "final_by_synthesis": final_by_synthesis,
        "final_by_weighted": final_by_weighted
    }


# ============================================================================
# 8. RAG SPECIFIC EXAMPLE - Compare RAG with two contexts
# ============================================================================

def aggregate_rag_responses(user_query, context1, context2, google_llm=None, 
                           context1_name="Context 1", context2_name="Context 2"):
    """
    Compare two RAG responses with different contexts
    Similar to your PDF summarization use case
    
    Args:
        user_query (str): Question to ask
        context1 (str): First context/knowledge base
        context2 (str): Second context/knowledge base
        google_llm: Language model to use
        context1_name (str): Name of first context
        context2_name (str): Name of second context
    
    Returns:
        dict: Dictionary with individual and synthesized responses
    """
    
    if google_llm is None:
        google_llm, _, _ = initialize_llms()
    
    print("\n" + "="*80)
    print("RAG RESPONSE AGGREGATION")
    print("="*80)
    
    system_prompt = """
    You are a helpful AI assistant. Answer the question using the provided context.
    If the context doesn't contain the answer, say so clearly.
    Be accurate and cite relevant parts of the context when applicable.
    """
    
    # Create RAG prompts for both contexts
    rag_prompt = ChatPromptTemplate.from_template("""
    {system_prompt}
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)
    
    # Get response with context1
    print(f"\nGenerating response with {context1_name}...")
    chain1 = rag_prompt | google_llm | StrOutputParser()
    response1 = chain1.invoke({
        "system_prompt": system_prompt,
        "context": context1,
        "question": user_query
    })
    print(f"✓ Response with {context1_name}:\n{response1}")
    
    # Get response with context2
    print(f"\nGenerating response with {context2_name}...")
    chain2 = rag_prompt | google_llm | StrOutputParser()
    response2 = chain2.invoke({
        "system_prompt": system_prompt,
        "context": context2,
        "question": user_query
    })
    print(f"✓ Response with {context2_name}:\n{response2}")
    
    # Synthesize both responses
    print("\nSynthesizing responses...")
    synthesis_prompt = ChatPromptTemplate.from_template("""
    Synthesize these two answers into one comprehensive response:
    
    Answer from {context1_name}:
    {response1}
    
    Answer from {context2_name}:
    {response2}
    
    Original Question: {question}
    
    Instructions:
    1. Combine the best insights from both answers
    2. Avoid redundancy
    3. Provide a complete, well-organized answer
    4. Maintain accuracy and clarity
    
    Provide a single synthesized answer that is better and more complete than both.
    """)
    
    chain_synthesis = synthesis_prompt | google_llm | StrOutputParser()
    final_response = chain_synthesis.invoke({
        "response1": response1,
        "response2": response2,
        "question": user_query,
        "context1_name": context1_name,
        "context2_name": context2_name
    })
    
    print(f"✓ Synthesized Final Response:\n{final_response}")
    
    return {
        "query": user_query,
        f"response_from_{context1_name}": response1,
        f"response_from_{context2_name}": response2,
        "synthesized_final_response": final_response
    }


# ============================================================================
# 9. UTILITY FUNCTIONS
# ============================================================================

def save_aggregation_result(result, filename="aggregation_result.json"):
    """
    Save aggregation results to a JSON file
    
    Args:
        result (dict): Result dictionary from aggregation
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Results saved to {filename}")


def print_comparison(result):
    """
    Print a formatted comparison of all aggregation methods
    
    Args:
        result (dict): Result dictionary from complete_aggregation_workflow
    """
    print("\n" + "="*80)
    print("COMPARISON OF ALL AGGREGATION METHODS")
    print("="*80)
    
    print(f"\nOriginal Question:\n{result['question']}\n")
    
    print("-"*80)
    print("Response 1 (First LLM):")
    print("-"*80)
    print(result['response1'][:300] + "..." if len(result['response1']) > 300 else result['response1'])
    
    print("\n" + "-"*80)
    print("Response 2 (Second LLM):")
    print("-"*80)
    print(result['response2'][:300] + "..." if len(result['response2']) > 300 else result['response2'])
    
    print("\n" + "-"*80)
    print("Final Response by SELECTION:")
    print("-"*80)
    print(result['final_by_selection'][:300] + "..." if len(result['final_by_selection']) > 300 else result['final_by_selection'])
    
    print("\n" + "-"*80)
    print("Final Response by SYNTHESIS:")
    print("-"*80)
    print(result['final_by_synthesis'][:300] + "..." if len(result['final_by_synthesis']) > 300 else result['final_by_synthesis'])
    
    print("\n" + "-"*80)
    print("Final Response by WEIGHTED COMBINATION:")
    print("-"*80)
    print(result['final_by_weighted'][:300] + "..." if len(result['final_by_weighted']) > 300 else result['final_by_weighted'])


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("MULTI-LLM RESPONSE AGGREGATION - REFERENCE GUIDE")
    print("="*80)
    
    # Example 1: Complete workflow with generic question
    question = "What is the importance of machine learning in modern applications?"
    
    try:
        result = complete_aggregation_workflow(question)
        
        if result:
            print_comparison(result)
            save_aggregation_result(result)
        
    except Exception as e:
        print(f"✗ Error in complete workflow: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: RAG specific aggregation
    print("\n\n")
    print("="*80)
    print("EXAMPLE 2: RAG RESPONSE AGGREGATION")
    print("="*80)
    
    try:
        google_llm, _, _ = initialize_llms()
        
        context1 = """
        Machine Learning is a subset of Artificial Intelligence that enables systems 
        to learn and improve from experience without being explicitly programmed.
        It uses algorithms to find patterns and make predictions without explicit programming.
        """
        
        context2 = """
        ML applications include recommendation systems, image recognition, and predictive analytics.
        It powers autonomous vehicles, medical diagnosis systems, and natural language processing.
        Major companies use ML for fraud detection, customer service, and personalization.
        """
        
        rag_result = aggregate_rag_responses(
            "What is machine learning and what are its key applications?",
            context1,
            context2,
            google_llm,
            context1_name="ML Definition",
            context2_name="ML Applications"
        )
        
        save_aggregation_result(rag_result, "rag_aggregation_result.json")
        
    except Exception as e:
        print(f"✗ Error in RAG aggregation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ EXECUTION COMPLETE")
    print("="*80)
    print("\nFor future reference, this module provides:")
    print("1. initialize_llms() - Set up multiple LLMs")
    print("2. get_dual_responses() - Get responses from two LLMs")
    print("3. judge_responses() - Evaluate responses")
    print("4. aggregate_by_selection() - Choose best response")
    print("5. aggregate_by_synthesis() - Merge responses")
    print("6. aggregate_by_weighted_combination() - Weighted aggregation")
    print("7. complete_aggregation_workflow() - Full pipeline")
    print("8. aggregate_rag_responses() - RAG-specific aggregation")
