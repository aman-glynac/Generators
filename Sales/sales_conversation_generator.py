import json
import random
import time
import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from groq import Groq

# Configuration and Data Models
@dataclass
class ConversationConfig:
    num_conversations: int = 100
    min_messages: int = 12
    max_messages: int = 15
    delay_between_requests: float = 1.0

class AIProvider(ABC):
    """Abstract base class for AI providers to enable easy switching"""
    
    @abstractmethod
    def generate_conversation(self, prompt: str) -> str:
        pass

class GroqProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate_conversation(self, prompt: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.8,
                max_tokens=3000,
                top_p=0.9
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None

# Uncomment and modify these classes when switching to other providers
"""
class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_conversation(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None

class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate_conversation(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating conversation: {e}")
            return None
"""

class B2BConversationGenerator:
    def __init__(self, ai_provider: AIProvider, config: ConversationConfig = None):
        self.ai_provider = ai_provider
        self.config = config or ConversationConfig()
        self.conversations = []
        
        # Industry and scenario data for variety
        self.industries = [
            "Software/SaaS", "Manufacturing", "Healthcare", "Financial Services", 
            "Retail/E-commerce", "Real Estate", "Construction", "Marketing/Advertising",
            "Logistics/Supply Chain", "Energy", "Education", "Telecommunications",
            "Food & Beverage", "Automotive", "Consulting", "Insurance"
        ]
        
        self.company_sizes = [
            "startup (10-50 employees)", "small business (50-200 employees)",
            "mid-market (200-1000 employees)", "enterprise (1000+ employees)"
        ]
        
        self.sales_scenarios = [
            "Initial cold outreach", "Follow-up after demo", "Pricing negotiation",
            "Objection handling", "Contract renewal", "Upselling existing client",
            "Cross-selling opportunity", "Competitive displacement", "Budget approval",
            "Technical requirements discussion", "Implementation timeline planning",
            "Stakeholder alignment", "Proof of concept discussion", "Risk assessment"
        ]
        
        self.pain_points = [
            "operational inefficiency", "high costs", "poor customer experience",
            "scalability challenges", "compliance issues", "data security concerns",
            "manual processes", "lack of visibility", "integration problems",
            "talent retention", "market competition", "regulatory changes"
        ]

    def create_advanced_prompt(self) -> str:
        """
        Advanced prompt using multiple prompt engineering techniques:
        - Chain of thought reasoning
        - Few-shot examples
        - Role definition
        - Constraint specification
        - Context setting
        - Output formatting
        """
        
        # Randomly select scenario parameters for variety
        industry = random.choice(self.industries)
        company_size = random.choice(self.company_sizes)
        scenario = random.choice(self.sales_scenarios)
        pain_point = random.choice(self.pain_points)
        
        prompt = f"""
# ROLE AND CONTEXT
You are an expert B2B sales conversation simulator. Your task is to generate a realistic, authentic B2B sales conversation that mirrors real-world business interactions.

# CONVERSATION PARAMETERS
- Industry: {industry}
- Company Size: {company_size}
- Scenario: {scenario}
- Primary Pain Point: {pain_point}
- Conversation Length: {random.randint(self.config.min_messages, self.config.max_messages)} messages

# CHARACTERS INSTRUCTIONS
Create 2-3 realistic business professionals with:
- Authentic names and job titles appropriate for the industry
- Distinct communication styles and personalities
- Realistic business concerns and motivations
- Industry-specific knowledge and terminology

# CONVERSATION QUALITY REQUIREMENTS
The conversation MUST include:
1. Natural business dialogue with realistic interruptions and clarifications
2. Industry-specific terminology and challenges
3. Authentic objections and responses
4. Progressive conversation flow (introduction → exploration → presentation → handling concerns → next steps)
5. Realistic business context (budget concerns, timeline pressures, stakeholder involvement)
6. Professional yet conversational tone
7. Specific details (numbers, dates, processes, competitors when relevant)

# REALISM TECHNIQUES TO APPLY
- Include small talk and relationship building moments
- Add realistic interruptions or schedule constraints
- Reference specific tools, processes, or industry standards
- Include follow-up questions and clarifications
- Show genuine interest and engagement from both parties
- Add realistic concerns about implementation, costs, or timing

# OUTPUT FORMAT REQUIREMENTS
Format each message as: #[Name]: [Message content]
Example:
#Sarah Chen: Hi Michael, thanks for taking the time to speak with me today...
#Michael Rodriguez: Of course, Sarah. I'm interested to learn more about how your solution...

# FEW-SHOT EXAMPLES FOR REFERENCE

## Example 1 - Software Industry Initial Outreach:
#Jennifer Walsh: Hi David, I'm Jennifer from CloudSecure Solutions. I noticed your company recently expanded into three new markets - congratulations! I imagine that's brought some interesting security challenges with managing data across multiple regions?

#David Kumar: Thanks Jennifer. Yes, it's been quite a journey. We're definitely feeling some growing pains, especially around ensuring consistent security protocols across all our locations. What exactly does CloudSecure do?

#Jennifer Walsh: We specialize in helping mid-market companies like yours maintain enterprise-level security without the enterprise complexity. I saw on your LinkedIn that you mentioned concerns about compliance audits - is that something keeping you up at night?

## Example 2 - Manufacturing Follow-up:
#Robert Thompson: Hi Lisa, thanks for the demo last week. I've had a chance to discuss it with my operations team, and we have some questions about integration with our existing ERP system.

#Lisa Martinez: Absolutely, Robert. Integration is always a key concern. What specific aspects of the ERP integration are your team most concerned about?

#Robert Thompson: Well, we're running SAP, and we've had some bad experiences with third-party tools not playing well with our existing workflows...

# TASK EXECUTION
Now generate a complete, realistic B2B sales conversation following all the above requirements. Make it authentic, engaging, and true to the specified industry and scenario.

Remember: This should read like a real conversation between real business professionals, not a scripted sales pitch.
"""
        return prompt

    def parse_conversation(self, raw_conversation: str) -> List[Dict[str, str]]:
        """Parse the generated conversation into structured format"""
        messages = []
        lines = raw_conversation.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and line.startswith('#'):
                # Extract speaker and message
                try:
                    colon_index = line.find(':', 1)  # Find first colon after #
                    if colon_index != -1:
                        speaker = line[1:colon_index].strip()
                        message = line[colon_index + 1:].strip()
                        if speaker and message:
                            messages.append({
                                'speaker': speaker,
                                'message': message
                            })
                except Exception as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
        
        return messages

    def validate_conversation(self, messages: List[Dict[str, str]]) -> bool:
        """Validate that the conversation meets quality requirements"""
        if len(messages) < self.config.min_messages:
            return False
        
        # Check for speaker variety
        speakers = set(msg['speaker'] for msg in messages)
        if len(speakers) < 2:
            return False
        
        # Check message quality (basic length check)
        avg_length = sum(len(msg['message']) for msg in messages) / len(messages)
        if avg_length < 20:  # Very short messages might indicate poor quality
            return False
        
        return True

    def format_conversation_output(self, messages: List[Dict[str, str]], conversation_id: int) -> str:
        """Format conversation for final output"""
        output = f"\n{'='*60}\nCONVERSATION #{conversation_id}\n{'='*60}\n"
        
        for msg in messages:
            output += f"#{msg['speaker']}: {msg['message']}\n"
        
        return output

    def generate_single_conversation(self, attempt: int = 1) -> Dict[str, Any]:
        """Generate a single conversation with retry logic"""
        max_attempts = 3
        
        for attempt_num in range(max_attempts):
            try:
                prompt = self.create_advanced_prompt()
                raw_conversation = self.ai_provider.generate_conversation(prompt)
                
                if not raw_conversation:
                    continue
                
                messages = self.parse_conversation(raw_conversation)
                
                if self.validate_conversation(messages):
                    return {
                        'id': attempt,
                        'messages': messages,
                        'raw_output': raw_conversation,
                        'success': True
                    }
                else:
                    print(f"Conversation {attempt} failed validation (attempt {attempt_num + 1})")
                    
            except Exception as e:
                print(f"Error generating conversation {attempt} (attempt {attempt_num + 1}): {e}")
        
        return {'id': attempt, 'success': False, 'error': 'Failed after maximum attempts'}

    def generate_all_conversations(self):
        """Generate all conversations with progress tracking"""
        print(f"Starting generation of {self.config.num_conversations} B2B sales conversations...")
        print(f"Target: {self.config.min_messages}-{self.config.max_messages} messages per conversation")
        print("-" * 60)
        
        successful_conversations = 0
        failed_conversations = 0
        
        for i in range(1, self.config.num_conversations + 1):
            print(f"Generating conversation {i}/{self.config.num_conversations}...", end=" ")
            
            conversation = self.generate_single_conversation(i)
            
            if conversation['success']:
                self.conversations.append(conversation)
                successful_conversations += 1
                print(f"✓ ({len(conversation['messages'])} messages)")
            else:
                failed_conversations += 1
                print("✗ Failed")
            
            # Rate limiting
            if i < self.config.num_conversations:
                time.sleep(self.config.delay_between_requests)
        
        print("-" * 60)
        print(f"Generation complete!")
        print(f"Successful: {successful_conversations}")
        print(f"Failed: {failed_conversations}")
        print(f"Success rate: {(successful_conversations/self.config.num_conversations)*100:.1f}%")

    def save_conversations(self, filename: str = "b2b_conversations.txt"):
        """Save all conversations to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("B2B SALES CONVERSATIONS DATASET\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Conversations: {len(self.conversations)}\n")
            f.write("=" * 80 + "\n")
            
            for conv in self.conversations:
                formatted_output = self.format_conversation_output(
                    conv['messages'], 
                    conv['id']
                )
                f.write(formatted_output)
        
        print(f"Conversations saved to {filename}")

    def save_json_format(self, filename: str = "b2b_conversations.json"):
        """Save conversations in JSON format for further processing"""
        json_data = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_conversations': len(self.conversations),
                'config': {
                    'min_messages': self.config.min_messages,
                    'max_messages': self.config.max_messages
                }
            },
            'conversations': []
        }
        
        for conv in self.conversations:
            json_data['conversations'].append({
                'id': conv['id'],
                'messages': conv['messages'],
                'message_count': len(conv['messages']),
                'participants': list(set(msg['speaker'] for msg in conv['messages']))
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON data saved to {filename}")

    def display_sample_conversations(self, num_samples: int = 3):
        """Display sample conversations for preview"""
        print(f"\n{'='*60}")
        print(f"SAMPLE CONVERSATIONS (showing {min(num_samples, len(self.conversations))} out of {len(self.conversations)})")
        print(f"{'='*60}")
        
        for i, conv in enumerate(self.conversations[:num_samples]):
            formatted_output = self.format_conversation_output(
                conv['messages'], 
                conv['id']
            )
            print(formatted_output)


# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Set your API key as environment variable

if not GROQ_API_KEY:
    print("Please set your GROQ_API_KEY environment variable")
    print("Example: export GROQ_API_KEY='your_api_key_here'")
    return

# Initialize configuration
config = ConversationConfig(
    num_conversations=100,  # Change this number as needed
    min_messages=12,
    max_messages=15,
    delay_between_requests=1.0
)

# Initialize AI provider (easily switchable)
ai_provider = GroqProvider(GROQ_API_KEY, model="llama3-8b-8192")

# To switch providers, uncomment and modify:
# ai_provider = OpenAIProvider(OPENAI_API_KEY, model="gpt-3.5-turbo")
# ai_provider = GeminiProvider(GEMINI_API_KEY, model="gemini-pro")

# Initialize generator
generator = B2BConversationGenerator(ai_provider, config)

# Generate conversations
generator.generate_all_conversations()

# Display samples
generator.display_sample_conversations(3)

# Save results
generator.save_conversations("b2b_sales_conversations.txt")
generator.save_json_format("b2b_sales_conversations.json")

print("\nGeneration complete! Check the output files for all conversations.")
