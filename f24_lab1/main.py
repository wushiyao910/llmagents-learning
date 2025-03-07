from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import math
import re

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # This function takes in a restaurant name and returns the reviews for that restaurant.
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    restaurant_reviews = {}
    
    try:
        with open("restaurant-data.txt", "r") as file:
            for line in file:
                # Extract the restaurant name from the beginning of the line
                parts = line.strip().split('. ', 1)
                if len(parts) >= 2:
                    current_restaurant = parts[0]
                    review = parts[1]
                    
                    # Case-insensitive matching of restaurant names
                    if restaurant_name.lower() in current_restaurant.lower():
                        if current_restaurant not in restaurant_reviews:
                            restaurant_reviews[current_restaurant] = []
                        restaurant_reviews[current_restaurant].append(review)
    except Exception as e:
        print(f"Error reading restaurant data: {e}")
    
    return restaurant_reviews


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    
    if len(food_scores) == 0 or len(customer_service_scores) == 0 or len(food_scores) != len(customer_service_scores):
        return {restaurant_name: 0.000}
    
    N = len(food_scores)
    total_score = 0.0
    
    for i in range(N):
        food_score = food_scores[i]
        customer_service_score = customer_service_scores[i]
        total_score += math.sqrt(food_score**2 * customer_service_score) * (1 / (N * math.sqrt(125))) * 10
    
    # Format to include at least 3 decimal places
    return {restaurant_name: round(total_score, 3)}


def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    # Return a prompt for the data fetch agent to use to fetch reviews for a specific restaurant
    return f"""You are a data retrieval agent. Your task is to analyze the given restaurant query and determine the restaurant name that should be queried from our database.

Query: "{restaurant_query}"

Extract the restaurant name from the query. For example:
- If the query is "What is the overall score for Taco Bell?", the restaurant name is "Taco Bell"
- If the query is "How good is Subway as a restaurant?", the restaurant name is "Subway"
- If the query is "What would you rate In N Out?", the restaurant name is "In N Out"

After identifying the restaurant name, call the function "fetch_restaurant_data" with the restaurant name as an argument to retrieve the relevant reviews.

Function signature: fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]
"""


def get_review_analyzer_agent_prompt() -> str:
    # Return a prompt for the review analyzer agent
    return """You are a review analyzer agent. Your task is to analyze restaurant reviews and extract scores for food quality and customer service based on specific keywords.

For each review, you need to extract two scores:
1. food_score: the quality of food at the restaurant (score from 1-5)
2. customer_service_score: the quality of customer service at the restaurant (score from 1-5)

Use the following scoring criteria based on keywords:
- Score 1/5: awful, horrible, or disgusting
- Score 2/5: bad, unpleasant, or offensive
- Score 3/5: average, uninspiring, or forgettable
- Score 4/5: good, enjoyable, or satisfying
- Score 5/5: awesome, incredible, or amazing

Each review will contain exactly two of these keywords - one describing food and one describing customer service.

For example, in the review:
"The food at McDonald's was average, but the customer service was unpleasant. The uninspiring menu options were served quickly, but the staff seemed disinterested and unhelpful."

- Food keywords: "average", "uninspiring" -> food_score: 3
- Customer service keyword: "unpleasant" -> customer_service_score: 2

Analyze each review in the given list and extract these two scores for each review. Present your findings in a structured format with the restaurant name, list of food scores, and list of customer service scores.
"""


def get_scoring_agent_prompt() -> str:
    # Return a prompt for the scoring agent
    return """You are a scoring agent. Your task is to take the extracted food scores and customer service scores for a restaurant and calculate an overall score.

You should receive:
1. The restaurant name
2. A list of food scores (from 1-5) for each review
3. A list of customer service scores (from 1-5) for each review

Call the function "calculate_overall_score" with these three parameters to get the overall score for the restaurant.

Function signature: calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]

The function will return a dictionary with the restaurant name as the key and the calculated score as the value.
"""


# Do not modify the signature of the "main" function.
def main(user_query: str):
    # Setup the entrypoint agent
    entrypoint_agent_system_message = """You are a restaurant review analysis supervisor. You coordinate the process of analyzing restaurant reviews to provide scores for restaurants based on user queries.

Your workflow involves:
1. First, identify the restaurant name from the user's query using a data fetch agent
2. Then, fetch all reviews for that restaurant
3. Next, analyze each review to extract food and customer service scores using a review analyzer agent
4. Finally, calculate an overall score using a scoring agent

You'll work with other agents sequentially to complete this task, and your goal is to provide the final score to the user.
"""
    
    # LLM config for all agents
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o-mini", 
                "api_key": os.environ.get("OPENAI_API_KEY")
            }
        ]
    }
    
    # The main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent", 
        system_message=entrypoint_agent_system_message, 
        llm_config=llm_config
    )
    
    # Register the fetch_restaurant_data function for both LLM suggestion and execution
    entrypoint_agent.register_for_llm(
        name="fetch_restaurant_data", 
        description="Fetches the reviews for a specific restaurant."
    )(fetch_restaurant_data)
    
    entrypoint_agent.register_for_execution(
        name="fetch_restaurant_data"
    )(fetch_restaurant_data)
    
    # Create the data fetch agent
    data_fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message=get_data_fetch_agent_prompt(user_query),
        llm_config=llm_config
    )

    # Register the fetch_restaurant_data function for the data fetch agent
    data_fetch_agent.register_for_llm(
        name="fetch_restaurant_data",
        description="Fetches the reviews for a specific restaurant."
    )(fetch_restaurant_data)
    
    data_fetch_agent.register_for_execution(
        name="fetch_restaurant_data"
    )(fetch_restaurant_data)
    
    # Create the review analyzer agent
    review_analyzer_agent = ConversableAgent(
        "review_analyzer_agent",
        system_message=get_review_analyzer_agent_prompt(),
        llm_config=llm_config
    )
    
    # Create the scoring agent
    scoring_agent = ConversableAgent(
        "scoring_agent",
        system_message=get_scoring_agent_prompt(),
        llm_config=llm_config
    )
    
    # Register the calculate_overall_score function for the scoring agent
    scoring_agent.register_for_llm(
        name="calculate_overall_score",
        description="Calculates the overall score for a restaurant based on food and customer service scores."
    )(calculate_overall_score)
    
    scoring_agent.register_for_execution(
        name="calculate_overall_score"
    )(calculate_overall_score)
    
    # Initiate the sequential chats
    result = entrypoint_agent.initiate_chats(
        [
            # First chat: fetch restaurant data
            {
                "recipient": data_fetch_agent,
                "message": f"I need to analyze reviews for a restaurant based on this query: '{user_query}'. Please identify the restaurant name and suggest how to fetch its reviews.",
                "summary_method": "last_msg",
            },
            # Second chat: analyze reviews
            {
                "recipient": review_analyzer_agent,
                "message": "Based on the fetched restaurant reviews, analyze each review to extract food scores and customer service scores.",
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "reflection_prompt": "Extract all the restaurant reviews that were fetched along with the restaurant name."
                }
            },
            # Third chat: calculate overall score
            {
                "recipient": scoring_agent,
                "message": "Based on the analyzed reviews, calculate the overall score for the restaurant.",
                "summary_method": "reflection_with_llm",
                "summary_args": {
                    "reflection_prompt": "Extract the restaurant name, list of food scores, and list of customer service scores from the analysis."
                }
            }
        ]
    )

# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])