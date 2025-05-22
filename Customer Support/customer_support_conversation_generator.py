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

class CustomerSupportConversationGenerator:
    def __init__(self, ai_provider: AIProvider, config: ConversationConfig = None):
        self.ai_provider = ai_provider
        self.config = config or ConversationConfig()
        self.conversations = []
        
        # Industry and scenario data for variety
        self.industries = [
            "E-commerce/Retail", "SaaS/Software", "Banking/Financial", "Telecommunications", 
            "Healthcare", "Insurance", "Travel/Hospitality", "Food Delivery", "Streaming Services",
            "Gaming", "Education/EdTech", "Real Estate", "Utilities", "Transportation/Rideshare",
            "Social Media", "Cloud Services", "Cybersecurity", "IoT/Smart Home"
        ]
        
        self.company_types = [
            "online marketplace", "subscription service", "mobile app", "web platform",
            "traditional retailer", "fintech startup", "enterprise software", "consumer electronics"
        ]
        
        self.support_scenarios = [
            "Technical issue troubleshooting", "Account access problems", "Billing and payment disputes",
            "Product defect or warranty claim", "Service outage or downtime", "Feature request or feedback",
            "Subscription cancellation or changes", "Data privacy concerns", "Integration problems",
            "Performance issues", "Security incident report", "Refund or return request",
            "Onboarding assistance", "Upgrade or downgrade support", "Third-party integration help",
            "Compliance or legal questions", "Account recovery", "Bug report follow-up"
        ]
        
        self.issue_types = [
            "login failures", "payment processing errors", "slow performance", "missing features",
            "data synchronization issues", "mobile app crashes", "email delivery problems",
            "API rate limiting", "browser compatibility", "network connectivity", "file upload failures",
            "notification not working", "search functionality broken", "reporting inaccuracies"
        ]
        
        self.customer_personas = [
            "tech-savvy power user", "frustrated small business owner", "elderly customer needing patience",
            "impatient executive", "detailed-oriented analyst", "casual consumer", "developer integrating APIs",
            "non-technical user", "compliance officer", "budget-conscious startup", "enterprise administrator"
        ]
        
        self.support_agent_types = [
            "junior support agent", "senior technical specialist", "escalation manager", 
            "billing specialist", "security expert", "product specialist"
        ]

    def create_advanced_prompt(self) -> str:
        """
        Advanced prompt using multiple prompt engineering techniques:
        - Chain of thought reasoning
        - Few-shot examples with multiple scenarios
        - Role definition with personality traits
        - Constraint specification
        - Context setting with realistic details
        - Output formatting requirements
        - Emotional intelligence guidelines
        """
        
        # Randomly select scenario parameters for variety
        industry = random.choice(self.industries)
        company_type = random.choice(self.company_types)
        scenario = random.choice(self.support_scenarios)
        issue_type = random.choice(self.issue_types)
        customer_persona = random.choice(self.customer_personas)
        agent_type = random.choice(self.support_agent_types)
        conversation_length = random.randint(self.config.min_messages, self.config.max_messages)
        
        # Add probability for multi-agent scenarios
        include_supervisor = random.random() < 0.3  # 30% chance
        include_specialist = random.random() < 0.2  # 20% chance
        
        prompt = f"""
# ROLE AND CONTEXT
You are an expert customer support conversation simulator. Your task is to generate a realistic, authentic customer support interaction that mirrors real-world customer service scenarios with natural human behavior, emotions, and problem-solving patterns.

# CONVERSATION PARAMETERS
- Industry: {industry}
- Company Type: {company_type}
- Support Scenario: {scenario}
- Primary Issue: {issue_type}
- Customer Profile: {customer_persona}
- Primary Agent Type: {agent_type}
- Conversation Length: {conversation_length} messages
- Multiple Agents: {"Yes (include supervisor/specialist)" if include_supervisor or include_specialist else "No (single agent)"}

# CHARACTER DEVELOPMENT REQUIREMENTS
Create realistic characters with:

**Customer Character:**
- Authentic name and background fitting the persona
- Realistic emotional state (frustrated, confused, urgent, patient, etc.)
- Appropriate technical knowledge level
- Genuine business or personal context
- Natural communication style with personality quirks

**Support Agent(s):**
- Professional names and realistic experience levels
- Distinct communication styles and expertise areas
- Appropriate empathy and problem-solving approaches
- Company-specific knowledge and procedures
- Professional yet human personality traits

# CONVERSATION QUALITY REQUIREMENTS
The conversation MUST demonstrate:

1. **Authentic Emotional Journey:**
   - Customer starts with genuine frustration, confusion, or concern
   - Emotional evolution throughout the interaction
   - Agent shows appropriate empathy and professionalism
   - Natural tension and resolution patterns

2. **Realistic Problem-Solving Process:**
   - Step-by-step troubleshooting or investigation
   - Multiple solution attempts (some may fail initially)
   - Information gathering and clarification
   - Escalation or specialist involvement when needed
   - Follow-up and confirmation steps

3. **Natural Dialogue Patterns:**
   - Realistic interruptions and clarifications
   - "Thank you" and "please" usage
   - Small acknowledgments ("I see", "Got it", "Okay")
   - Natural pauses and thinking moments
   - Realistic typing delays or "one moment please"

4. **Industry-Specific Details:**
   - Appropriate technical terminology
   - Company-specific processes and policies
   - Realistic system names, error codes, or reference numbers
   - Industry-standard troubleshooting steps
   - Relevant compliance or security considerations

5. **Professional Support Elements:**
   - Ticket or case number references
   - Screen sharing or diagnostic requests
   - Documentation or knowledge base references
   - Follow-up scheduling or email confirmations
   - Satisfaction surveys or feedback requests

# REALISM ENHANCEMENT TECHNIQUES
Apply these to increase authenticity:

- Include realistic delays ("Let me check that for you...")
- Add genuine confusion moments ("I'm not quite following...")
- Show human elements (apologizing for delays, showing appreciation)
- Include system limitations ("Our system shows...", "I need to escalate this...")
- Add realistic customer context ("I'm traveling", "My team is waiting", "This is urgent")
- Show knowledge gaps and learning ("I haven't seen this before", "Let me research...")
- Include genuine relief/satisfaction moments when issues resolve

# OUTPUT FORMAT REQUIREMENTS
Format each message as: #[Full Name]: [Complete message content]

Rules:
- Use realistic full names appropriate for the industry/region
- Include complete, natural sentences
- Show emotional undertones appropriately
- Include realistic business context when relevant

# COMPREHENSIVE FEW-SHOT EXAMPLES

## Example 1 - E-commerce Technical Issue:
#Sarah Mitchell: Hi there, I'm having a really frustrating issue with your checkout process. Every time I try to complete my order, it just spins and then gives me an error message saying "Payment processing failed" but my bank says the charge went through. This has happened three times now and I'm worried about being charged multiple times.

#David Rodriguez: Hi Sarah, I'm really sorry to hear about this frustrating experience with our checkout process. That definitely sounds concerning, especially with the potential multiple charges. Let me help you resolve this right away. Can you please provide me with your email address associated with the account so I can look up your recent order attempts?

#Sarah Mitchell: It's sarah.mitchell.designs@gmail.com. I was trying to order those ceramic planters that are on sale - order total should be around $247. The error happens right after I click "Complete Purchase."

#David Rodriguez: Thank you, Sarah. I can see your account here and I do see three attempted transactions from today. The good news is that none of them actually completed on our end, so you shouldn't be charged. However, your bank might be showing pending authorizations that should drop off in 1-2 business days. Let me check our system logs to see what's causing the checkout failure.

## Example 2 - SaaS Login Issues:
#Michael Chen: Hello, I urgently need help accessing my account. I've been locked out since this morning and my team has a critical presentation in 2 hours. The system keeps saying "Invalid credentials" but I know my password is correct. I even tried resetting it twice but the reset emails aren't coming through.

#Jennifer Walsh: Hi Michael, I understand how urgent this is with your presentation coming up. Let's get you back into your account as quickly as possible. I can see you're having authentication issues. First, can you confirm the email address associated with your account? Sometimes people have multiple work emails and try logging in with the wrong one.

#Michael Chen: It should be m.chen@innovatetech.com - that's my primary work email. I've been using this account for over a year without issues. Could this be related to the maintenance window you had last night?

#Jennifer Walsh: Yes, that email matches what I see in our system. And you're absolutely right - we did have a maintenance window last night that updated our authentication system. It looks like about 2% of users are experiencing similar issues. Let me manually refresh your account authentication while we're talking. This should take about 60 seconds.

## Example 3 - Multi-Agent Billing Dispute:
#Lisa Thompson: I need to speak with someone about my bill immediately. I've been charged $89.99 for a premium subscription that I never signed up for. I only use the basic free version of your service. This charge appeared on my credit card this morning and I need it reversed.

#Robert Kumar: Hi Lisa, I'm sorry to hear about this unexpected charge. I definitely understand your concern about being billed for a service you didn't sign up for. Let me pull up your account to investigate this. Can you provide me with the email address on your account?

#Lisa Thompson: It's lisa.thompson.marketing@gmail.com. I'm absolutely certain I never upgraded or clicked anything to upgrade. I've been very careful about that because I'm on a tight budget right now.

#Robert Kumar: I see your account here, Lisa. Looking at your billing history, it shows the premium upgrade was processed yesterday at 3:47 PM. However, I want to get our billing specialist involved to do a deeper investigation into how this upgrade was triggered. Let me bring in Amanda from our billing team who can look at the technical details of this transaction.

#Amanda Foster: Hi Lisa, this is Amanda from the billing department. Robert filled me in on the situation. I can see the premium upgrade in our logs, and I want to trace exactly how it was initiated. Sometimes these can be triggered by accidental clicks on mobile devices or through promotional emails. While I investigate, I'm going to process a full refund for you right now so you don't have to worry about the charge.

# ADVANCED EMOTIONAL INTELLIGENCE GUIDELINES

1. **Customer Emotional States to Portray:**
   - Initial frustration or confusion
   - Growing impatience if issues persist
   - Relief when progress is made
   - Gratitude when problems are resolved
   - Skepticism if solutions seem too simple
   - Urgency based on business impact

2. **Agent Professional Responses:**
   - Immediate acknowledgment of customer emotions
   - Appropriate apologies without over-apologizing
   - Clear explanation of next steps
   - Realistic timeframes and expectations
   - Proactive communication during delays
   - Genuine care for customer experience

3. **Conversation Flow Dynamics:**
   - Start with problem presentation
   - Move through information gathering
   - Include troubleshooting or investigation
   - Show progress updates and setbacks
   - Build toward resolution or escalation
   - End with confirmation and follow-up

# TASK EXECUTION
Generate a complete customer support conversation following all requirements above. The conversation should read like a real interaction between real people dealing with a genuine customer service issue.

Key Success Criteria:
- Natural, authentic dialogue
- Realistic problem-solving progression
- Appropriate emotional responses
- Industry-specific accuracy
- Professional yet human interaction
- Satisfying resolution or realistic escalation

Make this conversation feel completely authentic - as if someone recorded a real customer support session.
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
        if avg_length < 25:  # Customer support messages tend to be longer
            return False
        
        # Check for customer service patterns
        conversation_text = ' '.join(msg['message'].lower() for msg in messages)
        support_indicators = ['help', 'issue', 'problem', 'sorry', 'assist', 'resolve', 'fix']
        if not any(indicator in conversation_text for indicator in support_indicators):
            return False
        
        return True

    def format_conversation_output(self, messages: List[Dict[str, str]], conversation_id: int) -> str:
        """Format conversation for final output"""
        output = f"\n{'='*60}\nCUSTOMER SUPPORT CONVERSATION #{conversation_id}\n{'='*60}\n"
        
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
                        'success': True,
                        'participants': list(set(msg['speaker'] for msg in messages)),
                        'message_count': len(messages)
                    }
                else:
                    print(f"Conversation {attempt} failed validation (attempt {attempt_num + 1})")
                    
            except Exception as e:
                print(f"Error generating conversation {attempt} (attempt {attempt_num + 1}): {e}")
        
        return {'id': attempt, 'success': False, 'error': 'Failed after maximum attempts'}

    def generate_all_conversations(self):
        """Generate all conversations with progress tracking"""
        print(f"Starting generation of {self.config.num_conversations} customer support conversations...")
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
                participants = len(conversation['participants'])
                print(f"✓ ({conversation['message_count']} messages, {participants} participants)")
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
        
        # Additional statistics
        if self.conversations:
            avg_messages = sum(len(conv['messages']) for conv in self.conversations) / len(self.conversations)
            multi_agent_count = sum(1 for conv in self.conversations if len(conv['participants']) > 2)
            print(f"Average messages per conversation: {avg_messages:.1f}")
            print(f"Multi-agent conversations: {multi_agent_count} ({(multi_agent_count/len(self.conversations))*100:.1f}%)")

    def save_conversations(self, filename: str = "customer_support_conversations.txt"):
        """Save all conversations to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("CUSTOMER SUPPORT CONVERSATIONS DATASET\n")
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

    def save_json_format(self, filename: str = "customer_support_conversations.json"):
        """Save conversations in JSON format for further processing"""
        json_data = {
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_conversations': len(self.conversations),
                'config': {
                    'min_messages': self.config.min_messages,
                    'max_messages': self.config.max_messages
                },
                'statistics': {
                    'avg_messages_per_conversation': sum(len(conv['messages']) for conv in self.conversations) / len(self.conversations) if self.conversations else 0,
                    'multi_agent_conversations': sum(1 for conv in self.conversations if len(conv['participants']) > 2),
                    'unique_participants': len(set(participant for conv in self.conversations for participant in conv['participants']))
                }
            },
            'conversations': []
        }
        
        for conv in self.conversations:
            json_data['conversations'].append({
                'id': conv['id'],
                'messages': conv['messages'],
                'message_count': len(conv['messages']),
                'participants': conv['participants'],
                'participant_count': len(conv['participants'])
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
            if i < min(num_samples, len(self.conversations)) - 1:
                print(f"\n{'-'*40}\n")


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
generator = CustomerSupportConversationGenerator(ai_provider, config)

# Generate conversations
generator.generate_all_conversations()

# Display samples
generator.display_sample_conversations(3)

# Save results
generator.save_conversations("customer_support_conversations.txt")
generator.save_json_format("customer_support_conversations.json")

print("\nGeneration complete! Check the output files for all conversations.")
