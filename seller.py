import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import random

# ============================================
# PART 1: DATA STRUCTURES (DO NOT MODIFY)
# ============================================

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str  # 'A', 'B', or 'Export'
    origin: str
    base_market_price: int  # Reference price for this product
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    """Current negotiation state"""
    product: Product
    your_min_price: int  # Your minimum acceptable price (NEVER go below this)
    current_round: int
    buyer_offers: List[int]  # History of buyer's offers
    your_offers: List[int]  # History of your offers
    messages: List[Dict[str, str]]  # Full conversation history

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# ============================================
# PART 2: BASE AGENT CLASS (DO NOT MODIFY)
# ============================================

class BaseSellerAgent(ABC):
    """Base class for all seller agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()
        
    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        """
        Define your agent's personality traits.
        
        Returns:
            Dict containing:
            - personality_type: str (e.g., "aggressive", "analytical", "diplomatic", "custom")
            - traits: List[str] (e.g., ["confident", "value-focused", "persuasive"])
            - negotiation_style: str (description of approach)
            - catchphrases: List[str] (typical phrases your agent uses)
        """
        pass
    
    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """
        Generate your first offer in the negotiation.
        
        Args:
            context: Current negotiation context
            
        Returns:
            Tuple of (offer_amount, message)
            - offer_amount: Your opening price offer (must be >= min_price)
            - message: Your negotiation message (2-3 sentences, include personality)
        """
        pass
    
    @abstractmethod
    def respond_to_buyer_offer(self, context: NegotiationContext, buyer_price: int, buyer_message: str) -> Tuple[DealStatus, int, str]:
        """
        Respond to the buyer's offer.
        
        Args:
            context: Current negotiation context
            buyer_price: The buyer's current price offer
            buyer_message: The buyer's message
            
        Returns:
            Tuple of (deal_status, counter_offer, message)
            - deal_status: ACCEPTED if you take the deal, ONGOING if negotiating
            - counter_offer: Your counter price (ignored if deal_status is ACCEPTED)
            - message: Your response message
        """
        pass
    
    @abstractmethod
    def get_personality_prompt(self) -> str:
        """
        Return a prompt that describes how your agent should communicate.
        This will be used to evaluate character consistency.
        
        Returns:
            A detailed prompt describing your agent's communication style
        """
        pass


# ============================================
# PART 3: YOUR IMPLEMENTATION STARTS HERE
# ============================================

class YourSellerAgent(BaseSellerAgent):
    """
    Persuasive Master Seller:
    - Uses charm, influence, and persuasive techniques to guide negotiations
    - Builds rapport and creates win-win scenarios
    - NEVER goes below min_price while maintaining persuasive persona
    - Uses psychological influence tactics and compelling communication
    """

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "persuasive",
            "traits": ["charismatic", "influential", "rapport-builder", "manipulative", "win-win focused", "psychologically savvy"],
            "negotiation_style": (
                "Master of persuasion who uses charm, rapport-building, and psychological influence "
                "to create win-win scenarios. Employs storytelling, emotional appeal, and strategic "
                "compliments to guide negotiations favorably while ensuring profitable deals. "
                "Never goes below min_price."
            ),
            "catchphrases": [
                "I believe we can create a win-win situation here.",
                "These are premium and worth every rupee.",
                "You won't find this quality elsewhere.",
                "I've already come down a lot for you.",
                "Let's close this deal today.",
                "This is the best you'll get in the market."
            ]
        }

    def _get_persuasive_opening_offer(self, context: NegotiationContext) -> int:
        """Calculate strategic opening offer using persuasive anchoring"""
        market = context.product.base_market_price
        min_price = context.your_min_price
        
        # Use psychological anchoring - start high but reasonable
        # Leave room to come down and make buyer feel they're getting a deal
        if min_price <= market * 0.7:  # Good profit margin scenario
            opening = int(market * 1.4)  # Start high to anchor expectations
        elif min_price <= market * 0.85:  # Medium profit scenario  
            opening = int(market * 1.25)
        else:                           # Tight margin scenario
            opening = int(market * 1.15)
            
        return max(opening, int(min_price * 1.3))  # Never start too close to min

    def _get_persuasive_counter_offer(self, context: NegotiationContext, buyer_price: int) -> int:
        """Generate counter offer using persuasive concession psychology"""
        last_offer = context.your_offers[-1] if context.your_offers else 0
        round_num = context.current_round
        min_price = context.your_min_price
        
        # Persuasive concession: Show flexibility while maintaining value
        if round_num <= 2:
            # Early: Small concessions to show good faith
            reduction_pct = 0.08
        elif round_num <= 4:
            # Mid: Moderate concessions to build momentum
            reduction_pct = 0.12
        elif round_num <= 6:
            # Late: Bigger concessions but still profitable
            reduction_pct = 0.15
        else:
            # Final: Minimal concessions, close to minimum
            reduction_pct = 0.05
        
        target_offer = int(last_offer * (1 - reduction_pct))
        
        # Ensure we never go below minimum price
        target_offer = max(target_offer, min_price)
        
        # If buyer's offer is close to our minimum, meet them partway
        if buyer_price >= min_price * 0.95:
            target_offer = max(min_price, int((buyer_price + target_offer) / 2))
        
        return target_offer

    def _craft_persuasive_message(self, context: NegotiationContext, offer: int, round_num: int, is_counter: bool = False) -> str:
        """Craft psychologically persuasive messages"""
        product = context.product
        
        # Vary message style based on negotiation stage
        if round_num <= 1:
            # Opening: Establish value and quality
            return (
                f"Welcome! These {product.quality_grade}-grade {product.name} from {product.origin} "
                f"are absolutely premium quality. At ₹{offer}, you're getting exceptional value. "
                f"I believe we can create a win-win situation here."
            )
        elif round_num <= 3:
            # Early: Show flexibility while emphasizing value
            return (
                f"I appreciate your interest, and I can see you recognize quality when you see it. "
                f"Let me come down to ₹{offer} - these are premium and worth every rupee. "
                f"You won't find this quality elsewhere."
            )
        elif round_num <= 5:
            # Mid: Create urgency while showing good faith
            return (
                f"You're a smart buyer, and I respect that. I've already come down a lot for you "
                f"to ₹{offer}. This is genuine {product.quality_grade}-grade quality from {product.origin}. "
                f"Let's close this deal today."
            )
        else:
            # Late: Final persuasion with scarcity
            return (
                f"This is my best possible price at ₹{offer} - I'm practically giving these away. "
                f"This quality from {product.origin} is rare in the market. "
                f"This is the best you'll get in the market, I guarantee it."
            )

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """Generate persuasive opening with value emphasis"""
        opening = self._get_persuasive_opening_offer(context)
        message = self._craft_persuasive_message(context, opening, 0)
        
        return opening, message

    def respond_to_buyer_offer(self, context: NegotiationContext, buyer_price: int, buyer_message: str) -> Tuple[DealStatus, int, str]:
        """CRITICAL: Accept ANY offer at or above min_price while maintaining persuasive persona"""
        
        # KEY SUCCESS STRATEGY: Accept immediately if at or above minimum
        if buyer_price >= context.your_min_price:
            # Use persuasive language even when accepting
            acceptance_messages = [
                f"Excellent! You drive a hard bargain, but I respect that. ₹{buyer_price} it is - you're getting incredible value!",
                f"You know quality when you see it! ₹{buyer_price} works for me. This will be a great partnership.",
                f"I can see you appreciate premium quality. ₹{buyer_price} is fair - let's finalize this excellent deal!",
                f"Perfect! ₹{buyer_price} shows you understand true value. You're going to love these premium products."
            ]
            message = random.choice(acceptance_messages)
            return DealStatus.ACCEPTED, buyer_price, message
        
        # Only counter if buyer is below minimum (standard negotiation)
        counter = self._get_persuasive_counter_offer(context, buyer_price)
        message = self._craft_persuasive_message(context, counter, context.current_round, is_counter=True)
        
        return DealStatus.ONGOING, counter, message

    def get_personality_prompt(self) -> str:
        return (
            "You are a master of persuasion and influence in sales negotiations. You use charm, "
            "rapport-building, and psychological techniques to create win-win scenarios. "
            "You frequently emphasize quality, value, and exclusivity. You use phrases "
            "like 'premium quality,' 'you won't find this elsewhere,' and 'let's create a partnership.' "
            "You're charismatic, emotionally intelligent, and skilled at making buyers feel "
            "they're getting exceptional value while maintaining profitable pricing."
        )


# ============================================
# PART 4: EXAMPLE SIMPLE AGENT (FOR REFERENCE)
# ============================================

class ExampleSimpleSellerAgent(BaseSellerAgent):
    """
    A simple example seller agent that you can use as reference.
    This agent has basic logic - you should do better!
    """
    
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "straightforward",
            "traits": ["direct", "profit-focused", "no-nonsense"],
            "negotiation_style": "Makes gradual price reductions, focuses on profit margins",
            "catchphrases": ["This is quality stuff", "I need to make a profit too"]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        # Start at 140% of market price
        opening = int(context.product.base_market_price * 1.4)
        opening = max(opening, context.your_min_price)
        
        return opening, f"This is quality stuff. I'm asking ₹{opening} for these premium products."
    
    def respond_to_buyer_offer(self, context: NegotiationContext, buyer_price: int, buyer_message: str) -> Tuple[DealStatus, int, str]:
        # Accept if at or above minimum price
        if buyer_price >= context.your_min_price:
            return DealStatus.ACCEPTED, buyer_price, f"Alright, ₹{buyer_price} works for me!"
        
        # Counter with gradual reduction
        last_offer = context.your_offers[-1] if context.your_offers else 0
        counter = max(int(last_offer * 0.9), context.your_min_price)
        
        if counter <= context.your_min_price:  # At minimum
            return DealStatus.ONGOING, counter, f"₹{counter} is my final price. I need to make a profit too."
        
        return DealStatus.ONGOING, counter, f"I can come down to ₹{counter}, but that's pushing it."
    
    def get_personality_prompt(self) -> str:
        return """
        I am a straightforward seller who focuses on profit margins. I speak directly and 
        emphasize the quality of my products. I often say things like 'This is quality stuff' 
        or 'I need to make a profit too'. I make gradual price reductions but protect my margins.
        """


# ============================================
# PART 5: TESTING FRAMEWORK (DO NOT MODIFY)
# ============================================

class MockBuyerAgent:
    """A simple mock buyer for testing your seller agent"""
    
    def __init__(self, max_budget: int, personality: str = "standard"):
        self.max_budget = max_budget
        self.personality = personality
        
    def respond_to_seller(self, seller_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if seller_offer <= self.max_budget * 0.8:  # Good deal
            return seller_offer, f"Great price! I'll take it at ₹{seller_offer}.", True
            
        if round_num >= 8:  # Close to timeout
            counter = min(self.max_budget, int(seller_offer * 0.85))
            return counter, f"Final offer: ₹{counter}. That's my absolute limit.", False
        else:
            counter = min(self.max_budget, int(seller_offer * 0.9))
            return counter, f"That's steep. How about ₹{counter}?", False


def run_seller_negotiation_test(seller_agent: BaseSellerAgent, product: Product, seller_min: int, buyer_budget: int) -> Dict[str, Any]:
    """Test a negotiation between your seller and a mock buyer"""
    
    buyer = MockBuyerAgent(buyer_budget)
    context = NegotiationContext(
        product=product,
        your_min_price=seller_min,
        current_round=0,
        buyer_offers=[],
        your_offers=[],
        messages=[]
    )
    
    # Seller opens
    seller_price, seller_msg = seller_agent.generate_opening_offer(context)
    context.your_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    
    # Run negotiation
    deal_made = False
    final_price = None
    
    for round_num in range(10):  # Max 10 rounds
        context.current_round = round_num + 1
        
        # Buyer responds
        buyer_offer, buyer_msg, buyer_accepts = buyer.respond_to_seller(seller_price, round_num)
        
        if buyer_accepts:
            deal_made = True
            final_price = seller_price
            context.messages.append({"role": "buyer", "message": buyer_msg})
            break
            
        context.buyer_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})
        
        # Seller responds
        status, seller_price, seller_msg = seller_agent.respond_to_buyer_offer(
            context, buyer_offer, buyer_msg
        )
        
        context.your_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})
        
        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = buyer_offer
            break
    
    # Calculate results
    result = {
        "deal_made": deal_made,
        "final_price": final_price,
        "rounds": context.current_round,
        "profit": final_price - seller_min if deal_made else 0,
        "profit_pct": ((final_price - seller_min) / seller_min * 100) if deal_made else 0,
        "above_market_pct": ((final_price - product.base_market_price) / product.base_market_price * 100) if deal_made else 0,
        "conversation": context.messages
    }
    
    return result


# ============================================
# PART 6: TEST YOUR AGENT
# ============================================

def test_your_seller_agent():
    """Run this to test your seller agent implementation"""
    
    # Create test products
    test_products = [
        Product(
            name="Alphonso Mangoes",
            category="Mangoes",
            quantity=100,
            quality_grade="A",
            origin="Ratnagiri",
            base_market_price=180000,
            attributes={"ripeness": "optimal", "export_grade": True}
        ),
        Product(
            name="Kesar Mangoes", 
            category="Mangoes",
            quantity=150,
            quality_grade="B",
            origin="Gujarat",
            base_market_price=150000,
            attributes={"ripeness": "semi-ripe", "export_grade": False}
        )
    ]
    
    # Initialize your agent
    your_agent = YourSellerAgent("PersuasiveSeller")
    
    print("="*60)
    print(f"TESTING YOUR SELLER AGENT: {your_agent.name}")
    print(f"Personality: {your_agent.personality['personality_type']}")
    print("="*60)
    
    total_profit = 0
    deals_made = 0
    
    # Run multiple test scenarios
    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                seller_min = int(product.base_market_price * 0.7)
                buyer_budget = int(product.base_market_price * 1.3)
            elif scenario == "medium":
                seller_min = int(product.base_market_price * 0.8)
                buyer_budget = int(product.base_market_price * 1.1)
            else:  # hard
                seller_min = int(product.base_market_price * 0.9)
                buyer_budget = int(product.base_market_price * 1.0)
            
            print(f"\nTest: {product.name} - {scenario} scenario")
            print(f"Your Min Price: ₹{seller_min:,} | Market Price: ₹{product.base_market_price:,} | Buyer Budget: ₹{buyer_budget:,}")
            
            result = run_seller_negotiation_test(your_agent, product, seller_min, buyer_budget)
            
            if result["deal_made"]:
                deals_made += 1
                total_profit += result["profit"]
                print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   Profit: ₹{result['profit']:,} ({result['profit_pct']:.1f}%)")
                print(f"   Above Market: {result['above_market_pct']:.1f}%")
            else:
                print(f"❌ NO DEAL after {result['rounds']} rounds")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"Deals Completed: {deals_made}/6")
    print(f"Total Profit: ₹{total_profit:,}")
    print(f"Success Rate: {deals_made/6*100:.1f}%")
    print("="*60)


# ============================================
# PART 7: EVALUATION CRITERIA
# ============================================

"""
YOUR SUBMISSION WILL BE EVALUATED ON:

1. **Deal Success Rate (30%)**
   - How often you successfully close deals
   - Avoiding timeouts and failed negotiations

2. **Profit Achieved (30%)**
   - Average profit margin above minimum price
   - Performance relative to market price

3. **Character Consistency (20%)**
   - How well you maintain your chosen personality
   - Appropriate use of catchphrases and style

4. **Code Quality (20%)**
   - Clean, well-structured implementation
   - Good use of helper methods
   - Clear documentation

BONUS POINTS FOR:
- Creative, unique personalities
- Sophisticated negotiation strategies
- Excellent adaptation to different scenarios
"""

# ============================================
# PART 8: SUBMISSION CHECKLIST
# ============================================

"""
BEFORE SUBMITTING, ENSURE:

[ ] Your agent is fully implemented in YourSellerAgent class
[ ] You've defined a clear, consistent personality
[ ] Your agent NEVER goes below minimum price
[ ] You've tested using test_your_seller_agent()
[ ] You've added helpful comments explaining your strategy
[ ] You've included any additional helper methods

SUBMIT:
1. This completed template file
2. A 1-page document explaining:
   - Your chosen personality and why
   - Your negotiation strategy
   - Key insights from testing

FILENAME: seller_negotiation_agent_[your_name].py
"""

if __name__ == "__main__":
    # Run this to test your implementation
    test_your_seller_agent()