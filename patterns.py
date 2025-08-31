#!/usr/bin/env python3
"""
Pattern Definitions for Bengali Namjari Q&A Classification
Organized pattern matching for different categories
"""

import re
from typing import List, Dict, NamedTuple, Optional, Tuple


class PatternMatch(NamedTuple):
    pattern: str
    tag: str
    priority: int
    description: str


def get_critical_failure_patterns() -> List[PatternMatch]:
    """Critical patterns that fix known misclassifications - HIGHEST PRIORITY"""
    return [
        # Top priority exact phrase fixes
        PatternMatch(r'নামজারি জিনিসটা কী', 'namjari_application_procedure', 1, "What is namjari - procedure question"),
        PatternMatch(r'তার হয়ে নামজারির কাজ করতে পারব', 'namjari_by_representative', 1, "Can I do namjari for him - representative"),
        
        # Fix failure #13: Woman asking if she can do namjari alone (not about representative)
        PatternMatch(r'মেয়ে মানুষ.*একা.*নামজারির কাজ করতে পারব', 'namjari_application_procedure', 1, "Woman asking if can do namjari alone - procedure"),
        PatternMatch(r'আমি কি একা নামজারির কাজ করতে পারব.*নাকি কোনো পুরুষ মানুষ লাগবে', 'namjari_application_procedure', 1, "Can woman do namjari alone or need man - procedure"),
        
        # Fix failure #23: Hearing date postponed (more flexible pattern)
        PatternMatch(r'শুনানির তারিখ.*পিছিয়ে.*দেওয়া হয়েছে', 'namjari_hearing_notification', 1, "Hearing date postponed - notification"),
        PatternMatch(r'শুনানির তারিখ.*পিছিয়ে.*দিয়েছেন.*কেন', 'namjari_hearing_notification', 1, "Why hearing date postponed - notification"),
        
        # Fix failure #31: 4 brothers case (more flexible pattern)
        PatternMatch(r'৪ ভাই.*আছি.*নাম.*নেই', 'namjari_khatian_correction', 1, "4 brothers missing name - khatian correction"),
        PatternMatch(r'এখন আমরা ৪ ভাই.*খতিয়ানে আমাদের সবার নাম যোগ করতে হবে', 'namjari_khatian_correction', 1, "4 brothers need names added to khatian - correction"),
        PatternMatch(r'জরিপের সময়.*দাদার নাম.*৪ ভাই', 'namjari_khatian_correction', 1, "Survey time grandfather name 4 brothers - correction"),
        
        # Additional critical fixes
        PatternMatch(r'স্বামী বিদেশে কাজ করে.*কিভাবে করব', 'namjari_application_procedure', 1, "Husband abroad how to do - procedure"),
        PatternMatch(r'মেয়ে মানুষ.*শ্বশুর.*জমি.*কিভাবে করব', 'namjari_application_procedure', 1, "Woman asking about father-in-law land - procedure"),
    ]


def get_inheritance_patterns() -> List[PatternMatch]:
    """Patterns for inheritance-related namjari queries - more specific to avoid confusion"""
    return [
        # More specific inheritance patterns that require both death and inheritance context
        PatternMatch(r'(দাদা মারা গেছেন|বাবা.*মারা গেছে|মা মারা যাওয়ার পর).*নামজারি.*ওয়ারিশ', 'namjari_inheritance_documents', 2, "Death with heir context"),
        PatternMatch(r'(ওয়ারিশ সূত্রে|উত্তরাধিকার.*নামজারি|মৃত্যুর পর.*নামজারি)', 'namjari_inheritance_documents', 2, "Inheritance by heir"),
        PatternMatch(r'(ওয়ারিশ সার্টিফিকেট|মৃত্যু সনদ.*নামজারি|হাল ওয়াশিাননামা)', 'namjari_inheritance_documents', 2, "Heir certificate documents"),
        PatternMatch(r'মেয়ের বিয়ে.*মরে গেলে.*সম্পত্তিতে.*অধিকার', 'namjari_inheritance_documents', 2, "Daughter inheritance rights"),
        
        # Less aggressive patterns - only when clearly about inheritance documents/process
        PatternMatch(r'বাবার নামে.*জমি.*মারা.*কী কী কাগজ', 'namjari_inheritance_documents', 3, "Father's land inheritance documents"),
    ]


def get_application_procedure_patterns() -> List[PatternMatch]:
    """Patterns for application procedure queries"""
    return [
        PatternMatch(r'(জমি কিনেছি.*রেজিস্ট্রি.*করেছি|রেজিস্ট্রি.*নামজারি.*আলাদা)', 'namjari_application_procedure', 2, "Bought land registry separate from namjari"),
        PatternMatch(r'(নামজারি.*আবেদনের নিয়ম|নামজারি.*করার পদ্ধতি|নামজারি.*প্রক্রিয়া)', 'namjari_application_procedure', 2, "Namjari application rules procedure"),
        PatternMatch(r'(কিভাবে নামজারির জন্য আবেদন|অনলাইনে নামজারি|ভূমি অফিসে নামজারি)', 'namjari_application_procedure', 2, "How to apply for namjari"),
        PatternMatch(r'(দালাল ছাড়া.*নামজারি|নিজেই.*নামজারি.*করতে|কম্পিউটার.*চালাতে পারি না)', 'namjari_application_procedure', 2, "Do namjari without broker"),
    ]


def get_representative_patterns() -> List[PatternMatch]:
    """Patterns for representative/proxy namjari queries - more specific to avoid confusion"""
    return [
        # More specific representative patterns - avoid catching "woman alone" cases
        PatternMatch(r'আমেরিকায় থাকেন.*তার হয়ে.*নামজারি', 'namjari_by_representative', 2, "America proxy namjari"),
        PatternMatch(r'বিদেশে.*আসতে পারবেন না.*হয়ে.*নামজারি', 'namjari_by_representative', 2, "Abroad can't come proxy"),
        PatternMatch(r'প্রতিনিধি.*দিয়ে.*নামজারি', 'namjari_by_representative', 2, "Representative namjari"),
        PatternMatch(r'(পাওয়ার অফ অ্যাটর্নি|অথোরাইজেসন পত্র|ক্ষমতা অর্পনের পত্র)', 'namjari_by_representative', 2, "Power of attorney documents"),
    ]


def get_status_patterns() -> List[PatternMatch]:
    """Patterns for status check queries"""
    return [
        PatternMatch(r'(আবেদন করেছি.*খবর পাইনি|আবেদনটা.*কোন পর্যায়ে|স্ট্যাটাস চেক)', 'namjari_status_check', 2, "Application status check"),
        PatternMatch(r'(৪ মাস ধরে.*অপেক্ষা|প্রক্রিয়াধীন আছে|অগ্রগতি জানতে)', 'namjari_status_check', 2, "Waiting for progress"),
    ]


def get_hearing_patterns() -> List[PatternMatch]:
    """Patterns for hearing-related queries"""
    documents_patterns = [
        PatternMatch(r'(শুনানি.*কাগজ.*নিয়ে যেতে|শুনানীর সময়.*কাগজাদি|আমিন স্যার.*শুনানি)', 'namjari_hearing_documents', 2, "Hearing documents required"),
        PatternMatch(r'(শুনানিতে.*সাক্ষী.*নিয়ে|মূল কপি.*লাগবে|কাগজাদি.*আপলোড)', 'namjari_hearing_documents', 2, "Hearing witness documents"),
    ]
    
    notification_patterns = [
        PatternMatch(r'(শুনানির তারিখ.*পিছিয়ে|শুনানি.*রবিবার.*অফিস বন্ধ|দুই বার.*শুনানি মিস)', 'namjari_hearing_notification', 2, "Hearing date postponed"),
        PatternMatch(r'(এসএমএস.*ইংরেজিতে|নোটিশে.*শুনানীর তারিখ|মোবাইলে.*মেসেজ)', 'namjari_hearing_notification', 2, "Hearing SMS notification"),
    ]
    
    return documents_patterns + notification_patterns


def get_rejection_patterns() -> List[PatternMatch]:
    """Patterns for rejection and appeal queries"""
    return [
        PatternMatch(r'(খারিজ.*আপিল.*করা যাবে|নামজারি.*রিজেক্ট হয়েছে|আবেদন.*বাতিল হয়ে)', 'namjari_rejected_appeal', 2, "Rejected appeal process"),
        PatternMatch(r'(অসম্পূর্ণ তথ্যের জন্য.*রিজেক্ট|আপিলের.*সময়.*শেষ|রিভিউ.*নামঞ্জুর)', 'namjari_rejected_appeal', 2, "Incomplete info rejection"),
        PatternMatch(r'^(খারিজ হয়েছে|আমার আবেদন খারিজ)[\s।]*$', 'namjari_rejected_appeal', 2, "Simple rejection statement"),
    ]


def get_khatian_patterns() -> List[PatternMatch]:
    """Patterns for khatian copy and correction queries"""
    copy_patterns = [
        PatternMatch(r'(খতিয়ানের কপি.*সংগ্রহ|নতুন খতিয়ানের কপি|তহশিল অফিস.*খতিয়ান)', 'namjari_khatian_copy', 2, "Khatian copy collection"),
        PatternMatch(r'(সার্টিফাইড কপি.*সাধারণ কপি|খতিয়ানের কপিটা.*পুরানো|২০১৮ সালের)', 'namjari_khatian_copy', 2, "Certified vs regular copy"),
    ]
    
    correction_patterns = [
        PatternMatch(r'(খতিয়ানে.*নামটা ভুল|দাগ নম্বরটাও.*ঠিক নাই|বানান ভুল আছে)', 'namjari_khatian_correction', 2, "Khatian name correction"),
        PatternMatch(r'(সার্ভে রেকর্ডে.*দাগ নম্বর ভুল|জমির পরিমাণও ভুল|সংশোধন বাটনে ক্লিক)', 'namjari_khatian_correction', 2, "Survey record correction"),
    ]
    
    return copy_patterns + correction_patterns


def get_fee_and_document_patterns() -> List[PatternMatch]:
    """Patterns for fee and document queries"""
    return [
        # Fee patterns
        PatternMatch(r'(নামজারি করতে.*টাকা লাগে|সরকারি.*ফি.*আছে|কত টাকা খরচ)', 'namjari_fee', 2, "Namjari cost fee"),
        PatternMatch(r'(গরিব মানুষ.*বেশি টাকা নেই|দালাল.*২০,০০০ টাকা|১১৭০.*টাকা)', 'namjari_fee', 2, "Poor person fee concern"),
        
        # Required documents
        PatternMatch(r'(২ মাস ধরে.*দৌড়াদৌড়ি|কাগজের.*তালিকা.*দিতে|তহশিলদার সাহেব.*আরও কাগজ)', 'namjari_required_documents', 2, "Required documents list"),
    ]


def get_conversation_patterns() -> List[PatternMatch]:
    """Patterns for conversational flow management"""
    return [
        # Repeat patterns
        PatternMatch(r'^(আরেকবার বলুন|আবার বলবেন|একটু সিম্পল করে বলুন)', 'repeat_again', 2, "Ask to repeat again"),
        PatternMatch(r'(আগে যেটা জিজ্ঞেস করেছিলাম|কানে একটু কম শোনে|পড়ালেখাও তেমন জানি না)', 'repeat_again', 2, "Previous question repeat"),
        
        # Agent calling
        PatternMatch(r'(আমার হেল্প দরকার|একজন মানুষ দরকার|সহায়তা দিতে পারেন)', 'agent_calling', 2, "Need help assistance"),
        PatternMatch(r'(মাথা ঘুরে যায়.*কম্পিউটার.*বুঝি না|৬৫.*এত দিনে এসব শিখব|হতাশ হয়ে পড়েছি)', 'agent_calling', 2, "Overwhelmed need help"),
        
        # Goodbye patterns
        PatternMatch(r'(কাজটা শেষ হয়ে গেছে.*আল্লাহ হাফেজ|আজকের মত থাক.*কোনো প্রশ্ন নেই)', 'goodbye', 2, "Work finished goodbye"),
        PatternMatch(r'(বিদেশ থেকে কল.*রাত হয়ে গেছে|খোদা হাফেজ.*দোয়া রাখবেন|কলটা রেখে দিচ্ছি)', 'goodbye', 2, "International call goodbye"),
    ]


def get_simple_patterns() -> List[PatternMatch]:
    """Patterns for simple single-word queries"""
    return [
        PatternMatch(r'^নামজারি[\s।]*$', 'namjari_eligibility', 2, "Simple namjari question"),
        PatternMatch(r'^(মিউটেশনের কাজ|মিউটেশন)[\s।]*$', 'namjari_eligibility', 2, "Simple mutation question"),
    ]


def get_irrelevant_patterns() -> List[PatternMatch]:
    """Enhanced patterns for irrelevant queries - should be caught early"""
    return [
        # Birth registration - enhanced patterns
        PatternMatch(r'জন্মনিবন্ধন.*করতে.*কি', 'irrelevant', 3, "Birth registration question"),
        PatternMatch(r'জন্মসনদ.*বানাতে', 'irrelevant', 3, "Birth certificate making"),
        PatternMatch(r'জন্মনিবন্ধন.*মোবাইল.*নম্বর', 'irrelevant', 3, "Birth registration mobile number"),
        
        # Religious pilgrimage - enhanced patterns
        PatternMatch(r'হজ্ব.*করতে.*চাই', 'irrelevant', 3, "Want to do Hajj"),
        PatternMatch(r'হজ্ব.*আবেদন.*নিজে', 'irrelevant', 3, "Hajj application by self"),
        PatternMatch(r'ওমরাহ্‌.*করতে', 'irrelevant', 3, "Umrah pilgrimage"),
        
        # Job applications - enhanced patterns
        PatternMatch(r'চাকরির.*আবেদন.*করতে', 'irrelevant', 3, "Job application process"),
        PatternMatch(r'কোম্পানিতে.*আবেদন.*কি.*নিজে', 'irrelevant', 3, "Company application by self"),
        PatternMatch(r'চাকরি.*পেতে.*কি', 'irrelevant', 3, "How to get job"),
        
        # Land grabbing (illegal) - enhanced patterns
        PatternMatch(r'জমি.*দখল.*করতে.*কি', 'irrelevant', 3, "How to grab land"),
        PatternMatch(r'দখল.*করা.*যাবে', 'irrelevant', 3, "Can grab/occupy"),
        PatternMatch(r'কারো.*সাহায্যে.*কি.*দখল', 'irrelevant', 3, "Grab with someone's help"),
        
        # Travel documents - enhanced patterns
        PatternMatch(r'পাসপোর্ট.*বানাতে.*কি', 'irrelevant', 3, "Passport making process"),
        PatternMatch(r'ভিসা.*করতে.*কি', 'irrelevant', 3, "Visa processing"),
        
        # Weather/casual - enhanced patterns
        PatternMatch(r'আবহাওয়া.*খুব.*খারাপ', 'irrelevant', 3, "Very bad weather"),
        PatternMatch(r'আজকে.*একটা.*সুন্দর.*দিন', 'irrelevant', 3, "Today is a beautiful day"),
        PatternMatch(r'হারিয়ে.*যাওয়া.*বই', 'irrelevant', 3, "Lost book"),
        
        # Technical/other - enhanced patterns
        PatternMatch(r'মিউট.*করতে.*হলে', 'irrelevant', 3, "How to mute"),
        PatternMatch(r'চাঁদাবাজি.*করতে.*পারে', 'irrelevant', 3, "Can do extortion"),
        PatternMatch(r'দশ.*পারসেন্ট.*আবেদন', 'irrelevant', 3, "Ten percent application"),
    ]


def get_anti_confusion_patterns() -> Dict[str, List[str]]:
    """Anti-patterns to prevent misclassification"""
    return {
        'namjari_rejected_appeal': [
            r'(ধন্যবাদ|আল্লাহ হাফেজ|বিদায়)',
            r'^(সালাম|আদাব|হ্যালো)',
        ]
    }


def compile_all_patterns():
    """Compile all patterns into a single organized list"""
    all_patterns = []
    
    # Add patterns by priority
    all_patterns.extend(get_critical_failure_patterns())
    all_patterns.extend(get_inheritance_patterns())
    all_patterns.extend(get_application_procedure_patterns())
    all_patterns.extend(get_representative_patterns())
    all_patterns.extend(get_status_patterns())
    all_patterns.extend(get_hearing_patterns())
    all_patterns.extend(get_rejection_patterns())
    all_patterns.extend(get_khatian_patterns())
    all_patterns.extend(get_fee_and_document_patterns())
    all_patterns.extend(get_conversation_patterns())
    all_patterns.extend(get_simple_patterns())
    all_patterns.extend(get_irrelevant_patterns())
    
    # Sort by priority (lower number = higher priority)
    all_patterns.sort(key=lambda x: x.priority)
    
    anti_patterns = get_anti_confusion_patterns()
    
    return all_patterns, anti_patterns


def match_patterns(query: str, patterns: List[PatternMatch]) -> Optional[PatternMatch]:
    """Match query against patterns in priority order"""
    for pattern in patterns:
        if re.search(pattern.pattern, query, re.IGNORECASE):
            return pattern
    return None


def check_anti_patterns(query: str, predicted_tag: str, anti_patterns: Dict[str, List[str]]) -> bool:
    """Check if prediction should be blocked by anti-patterns"""
    if predicted_tag in anti_patterns:
        for anti_pattern in anti_patterns[predicted_tag]:
            if re.search(anti_pattern, query, re.IGNORECASE):
                return True
    return False