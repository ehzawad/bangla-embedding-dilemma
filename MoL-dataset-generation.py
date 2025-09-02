# -*- coding: utf-8 -*-
"""
Generate 1000 Bangla questions across 13 namjari tags using OpenAI gpt-5.
Final production version with streaming display and proper CSV output.
"""

import csv
import os
import time
from collections import defaultdict
from openai import OpenAI

# Configuration
MODEL = "gpt-5"
TARGET_ROWS = 1000
OUTPUT_DIR = "namjari_questions"
MAX_COMPLETION_TOKENS = 25000  # Reasonable limit per tag: ~15k reasoning + ~10k output

# Seed data for 13 tags
SEED_DATA = {
    "namjari_process": [
        "নামজারি সেবা কিভাবে পেতে পারি?",
        "নামজারি সেবা পেতে কি করতে হবে?", 
        "খারিজ করতে চাই, কি করতে হবে?",
        "অনলাইনে নামজারি করতে কি করতে হবে?",
        "মিউটেশন করতে হলে কি করতে হবে?"
    ],
    "namjari_application_procedure": [
        "আমি নামজারি করবো কি করতে হবে?",
        "নামজারি আবেদন কি নিজে করতে পারি।",
        "নামজারি করতে কি ভূমি অফিসে যাওয়া লাগে?",
        "আমি নিজে কি নামজারি করতে পারবো?",
        "কারো সাহায্যে কি নামজারি করা যাবে?"
    ],
    "namjari_registration": [
        "নামজারির জন্য নিবন্ধন করতে কি লাগে?",
        "নিবন্ধন করতে কি কি দরকার?",
        "নিবন্ধন করতে কি দলিল লাগে?",
        "নিবন্ধন করতে নিজের মোবাইল থাকতে হবে?",
        "নিবন্ধন করতে এনআইডি বা জন্মনিবন্ধন লাগে কি?"
    ],
    "namjari_by_representative": [
        "নামজারি আবেদন নিজে না করলে প্রতিনিধি দিয়ে করা যায় কিনা?",
        "আমি বিদেশে থাকি আমার ভাই বা আত্মীয় নামজারির আবেদন করতে পারে কিনা?",
        "প্রতিনিধি দিয়ে নামজারি করালে কি তার নামে জমি হয়ে যাবে?",
        "আমি থাকিনা, অন্য কেউ কি আমার নামজারি করতে পারবে?"
    ],
    "namjari_eligibility": [
        "নামজারি কখন করতে হয়?",
        "আমি জমি কিনেছি আমার কি নামজারি করা লাগবে?", 
        "আমি দলিল করে জমি পাইছি, নামজারি করা যাবে?",
        "আমার পিতা মারা গেছেন আমি তো জমি খাই, নামজারি করতে হবে কি?"
    ],
    "namjari_required_documents": [
        "নামজারি করতে কি কি দলিল লাগে?",
        "নামজারি করতে দলিল ছাড়া আর কি লাগে?",
        "নামজারি করতে ছবি আর এনআইডি লাগে কিনা?",
        "দলিলের ফটোকপি দিয়ে নামজারি হবে কিনা?"
    ],
    "namjari_inheritance_documents": [
        "ওয়ারিশ মতে নাম জারি করতে কিকি কাগজ লাঘে?",
        "ওয়ারিশ মতে অন লাইনে নামজারি করা যায় কিনা?",
        "আমার বাবা মারা গেছেন, এখন আমাদের নামে নামজারি করতে হবে কি?",
        "বন্টননামা নাই নামজারি করা যাবে কি?"
    ],
    "namjari_fee": [
        "নামজারির সরকারি ফি কত?",
        "নামজারি করতে কত টাকা লাগে?",
        "নামজারি করতে কত খরচ হবে?",
        "একাধিক আবেদনের জন্য কি বেশি টাকা লাগে?"
    ],
    "namjari_hearing_notification": [
        "নামজারির আবেদন দাখিলের পর শুনানীর জন্য কিভাবে জানবো?",
        "শুনানীর নোটিশ কিভাবে পাবো?",
        "নামজারির জন্য মোবাইল করে খোজ নেব কিনা?",
        "শুনানীর নোটিশের জন্য পোস্ট অফিসে খোঁজ নিতে হবে কি?"
    ],
    "namjari_hearing_documents": [
        "শুনানীর জন্য সব কাগজ কি নিয়ে যেতে হবে?",
        "শুনানীর জন্য কি কি কাগজপত্র নিয়ে যাবো?",
        "শুনানীর সময় কি আবেদনকারীকেই যেতে হবে?",
        "শুনানীতে সকল আবেদনকারীকে যেতে হবে নাতো?"
    ],
    "namjari_status_check": [
        "নামজারি মামলা কি অবস্থায় আছে কিভাবে জানবো?",
        "নামজারি আবেদন করেছি অনেকদিন কি আবস্থায় আছে জানবো কিভাবে?",
        "নামজারি মামলার স্ট্যাটাস কিভাবে জানতে পারবো?",
        "নামজারি মামলার অবস্থা জানার উপায় কি?"
    ],
    "namjari_rejected_appeal": [
        "নামজারি মামলা নামঞ্জুর হলে কি করবো?",
        "নামজারি মামলা নামঞ্জুর হয়েছে কোন আদালতে যাবো?", 
        "নামঞ্জুর হলে আবার কি আবেদন করবো?",
        "এসি (ল্যান্ড) শুনানী না নিয়ে নামঞ্জুর করেছেন?"
    ],
    "namjari_khatian_copy": [
        "নামজারি মঞ্জুর হয়েছে খতিয়ান কপি কিভাবে পেতে পারি?",
        "আমার নামজারি মঞ্জুর হয়েছে খতিয়ান কি আমি উঠাতে পারবো?",
        "অনলাইনে নামজারি করেছি আমার খতিয়ানের কপি কোথায় পাবো?",
        "নামজারি মঞ্জুর হলে খতিয়ানের প্রিন্ট কপি নিজে কি নিতে পারবো?"
    ],
    "namjari_khatian_correction": [
        "খতিয়ানে ভুল আছে সংশোধন করতে কি করতে হবে?",
        "খতিয়ানে নাম ভুল লেখা সংশোধন কিভাবে করবো?",
        "খতিয়ানে জমির পরিমাণ ভুল কিভাবে সংশোধন করবো?",
        "খতিয়ান সংশোধন করতে কি কি কাগজ লাগে?",
        "খতিয়ান সংশোধন করতে কত টাকা লাগে?"
    ]
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_question(text):
    """Basic cleaning of question text."""
    return text.strip().replace('"', '').replace("'", '')

def get_cross_tag_exclusions(current_tag):
    """Generate explicit exclusion rules with concrete examples to prevent cross-tag contamination."""
    
    # Build exclusion rules dynamically using actual SEED_DATA
    exclusion_rules = []
    
    if current_tag == 'namjari_process':
        exclusion_rules.extend([
            "❌ DON'T generate document questions like these (belongs to namjari_required_documents):",
        ])
        for example in SEED_DATA['namjari_required_documents'][:1]:  # Show only 1 example for efficiency
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate fee questions like these (belongs to namjari_fee):",
        ])
        for example in SEED_DATA['namjari_fee'][:1]:  # Show only 1 example for efficiency
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate inheritance questions like these (belongs to namjari_inheritance_documents):",
        ])
        for example in SEED_DATA['namjari_inheritance_documents'][:1]:  # Show only 1 example for efficiency
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate status questions like these (belongs to namjari_status_check):",
        ])
        for example in SEED_DATA['namjari_status_check'][:1]:  # Show only 1 example for efficiency
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.append("❌ ONLY generate general process questions like your examples!")
    
    elif current_tag == 'namjari_application_procedure':
        exclusion_rules.extend([
            "❌ DON'T generate document questions like these (belongs to namjari_required_documents):",
        ])
        for example in SEED_DATA['namjari_required_documents']:
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate representative questions like these (belongs to namjari_by_representative):",
        ])
        for example in SEED_DATA['namjari_by_representative']:
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate fee questions like these (belongs to namjari_fee):",
        ])
        for example in SEED_DATA['namjari_fee']:
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.append("❌ ONLY generate self-capability questions with 'নিজে', 'আমি নিজে' patterns!")
    
    elif current_tag == 'namjari_fee':
        exclusion_rules.extend([
            "❌ DON'T generate process questions like these (belongs to namjari_process):",
        ])
        for example in SEED_DATA['namjari_process']:
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate document questions like these (belongs to namjari_required_documents):",
        ])
        for example in SEED_DATA['namjari_required_documents']:
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.extend([
            "❌ DON'T generate hearing questions like these (belongs to namjari_hearing_notification):",
        ])
        for example in SEED_DATA['namjari_hearing_notification']:
            exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.append("❌ ONLY generate cost/fee questions with 'ফি', 'টাকা', 'খরচ', 'সরকারি'!")
    
    else:
        # For other tags, create a more generic exclusion pattern
        excluded_tags = [tag for tag in SEED_DATA.keys() if tag != current_tag][:3]  # Show top 3 most different tags
        for excluded_tag in excluded_tags:
            exclusion_rules.append(f"❌ DON'T generate questions like these (belongs to {excluded_tag}):")
            for example in SEED_DATA[excluded_tag][:1]:  # Show 1 example per excluded tag for efficiency
                exclusion_rules.append(f"   • \"{example}\"")
        
        exclusion_rules.append(f"❌ ONLY generate questions that fit {current_tag} domain!")
    
    return "\n".join(exclusion_rules) if exclusion_rules else "❌ Stay strictly within this tag's domain"

def analyze_question_patterns(seed_questions, tag):
    """Advanced pattern analysis capturing vocabulary, tone, context, and distinctions."""
    
    # Tag-specific vocabulary mapping
    tag_vocabularies = {
        'namjari_process': ['সেবা', 'খারিজ', 'মিউটেশন', 'অনলাইনে'],
        'namjari_application_procedure': ['নিজে', 'ভূমি অফিস', 'সাহায্যে', 'আমি নিজে'],
        'namjari_registration': ['নিবন্ধন', 'মোবাইল', 'এনআইডি', 'জন্মনিবন্ধন'],
        'namjari_by_representative': ['প্রতিনিধি', 'বিদেশে', 'ভাই', 'আত্মীয়'],
        'namjari_eligibility': ['কখন', 'জমি কিনেছি', 'পিতা মারা গেছেন', 'দলিল করে'],
        'namjari_required_documents': ['দলিল', 'ছবি', 'এনআইডি', 'ফটোকপি'],
        'namjari_inheritance_documents': ['ওয়ারিশ', 'বাবা মারা গেছেন', 'বন্টননামা'],
        'namjari_fee': ['ফি', 'টাকা', 'খরচ', 'সরকারি'],
        'namjari_hearing_notification': ['শুনানী', 'নোটিশ', 'মোবাইল করে', 'পোস্ট অফিস'],
        'namjari_hearing_documents': ['শুনানীর জন্য', 'কাগজপত্র', 'আবেদনকারী'],
        'namjari_status_check': ['মামলা', 'অবস্থায়', 'স্ট্যাটাস', 'জানবো কিভাবে'],
        'namjari_rejected_appeal': ['নামঞ্জুর', 'আদালত', 'এসি (ল্যান্ড)', 'আবার আবেদন'],
        'namjari_khatian_copy': ['মঞ্জুর', 'খতিয়ান কপি', 'উঠাতে', 'প্রিন্ট কপি'],
        'namjari_khatian_correction': ['ভুল', 'সংশোধন', 'নাম ভুল', 'জমির পরিমাণ']
    }
    
    # Tag-specific contexts and tones
    tag_contexts = {
        'namjari_process': 'General service inquiry tone - seeking basic information',
        'namjari_application_procedure': 'Self-capability concern tone - "আমি নিজে" patterns',
        'namjari_registration': 'Technical requirement tone - system setup focus',
        'namjari_by_representative': 'Delegation concern tone - long conditional questions',
        'namjari_eligibility': 'Situational qualification tone - life event contexts',
        'namjari_required_documents': 'Document-focused tone - practical requirements',
        'namjari_inheritance_documents': 'Emotional family tone - death/inheritance context',
        'namjari_fee': 'Cost-conscious tone - purely financial focus',
        'namjari_hearing_notification': 'Information-seeking tone - "কিভাবে জানবো" patterns',
        'namjari_hearing_documents': 'Preparation-focused tone - "কি নিয়ে যাবো" patterns',
        'namjari_status_check': 'Anxious follow-up tone - tracking progress',
        'namjari_rejected_appeal': 'Problem-solving tone - dealing with rejection',
        'namjari_khatian_copy': 'Success-phase tone - getting final documents',
        'namjari_khatian_correction': 'Error-fixing tone - correcting mistakes'
    }
    
    # Extract actual patterns from seed questions
    question_starters = []
    structures = []
    key_phrases = []
    
    for q in seed_questions:
        # Question starter analysis
        if q.startswith('কিভাবে'):
            question_starters.append('কিভাবে')
        elif q.startswith('কি '):
            question_starters.append('কি')
        elif q.startswith('কত'):
            question_starters.append('কত')
        elif q.startswith('কোথায়'):
            question_starters.append('কোথায়')
        elif q.startswith('আমি'):
            question_starters.append('আমি')
        
        # Structure analysis
        if 'কি করতে হবে' in q:
            structures.append('X কি করতে হবে?')
        if 'কিভাবে' in q and 'পারি' in q:
            structures.append('X কিভাবে Y পারি?')
        if 'কিভাবে' in q and 'পাবো' in q:
            structures.append('X কিভাবে Y পাবো?')
        if 'কত' in q and 'লাগে' in q:
            structures.append('কত X লাগে?')
        
        # Extract unique phrases
        for phrase in tag_vocabularies.get(tag, []):
            if phrase in q:
                key_phrases.append(phrase)
    
    analysis = f"""
ADVANCED PATTERN ANALYSIS FOR {tag.upper()}:

🎯 TAG-SPECIFIC CONTEXT: {tag_contexts.get(tag, 'Standard namjari context')}

🗣️ QUESTION STARTERS: {', '.join(set(question_starters)) if question_starters else 'Mixed'}
📝 SENTENCE STRUCTURES: {', '.join(set(structures)) if structures else 'Varied'}
💬 CRITICAL VOCABULARY: {', '.join(set(key_phrases)) if key_phrases else 'Standard'}
📊 EXPECTED VOCABULARY: {', '.join(tag_vocabularies.get(tag, ['নামজারি']))}

⚡ GENERATION RULES FOR THIS TAG:
1. MUST use the exact vocabulary: {', '.join(tag_vocabularies.get(tag, ['নামজারি']))}
2. MUST match the tone: {tag_contexts.get(tag, 'Standard')}
3. MUST follow structures: {', '.join(set(structures)) if structures else 'Same as examples'}
4. MUST start questions like examples: {', '.join(set(question_starters)) if question_starters else 'Varied starters'}

⭐ CRITICAL: This tag is DISTINCT from all others - maintain its unique vocabulary and context!"""
    
    return analysis

def display_generated_text_streaming(generated_text, tag):
    """Display generated text with streaming effect and return cleaned lines."""
    print(f"📺 Generated content for {tag}:")
    print("=" * 80)
    
    lines = generated_text.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        clean_line = clean_question(line)
        if len(clean_line) > 5 and any('\u0980' <= char <= '\u09FF' for char in clean_line):
            print(f"✨ {clean_line}")
            cleaned_lines.append(clean_line)
            time.sleep(0.1)  # Small delay for streaming effect
    
    print("=" * 80)
    return cleaned_lines

def generate_questions_for_tag(tag, seed_questions, target_count):
    """Generate questions for a tag and stream to file."""
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Create file and write header immediately
    filename = f"{tag}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    print(f"\n🏷️  Processing: {tag}")
    print(f"📁 Creating: {filepath}")
    print(f"🎯 Target: {target_count} questions")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['question', 'tag'])
        
        # Write seed questions first  
        print(f"📝 Writing {len(seed_questions)} seed questions...")
        for i, seed in enumerate(seed_questions, 1):
            clean_q = clean_question(seed)
            writer.writerow([clean_q, tag])
            f.flush()
            print(f"  {i:2d}. {clean_q}")
        
        seed_count = len(seed_questions)
        print(f"✅ Wrote {seed_count} seed questions")
        
        # Generate more if needed
        questions_needed = target_count - seed_count
        if questions_needed <= 0:
            print(f"✅ Target reached with seeds only")
            return filepath
        
        print(f"🤖 Generating {questions_needed} new questions...")
        
        # Create highly specific prompt for maximum similarity
        examples = "\n".join(seed_questions)
        
        # Analyze patterns in seed questions for this tag
        pattern_analysis = analyze_question_patterns(seed_questions, tag)
        
        prompt = f"""Generate {questions_needed} new Bengali questions that are 97-99% IDENTICAL in style, structure, and vocabulary to these examples:

{examples}

CRITICAL REQUIREMENTS - Follow these EXACTLY:
- Copy the exact sentence structures from examples above
- Use the same question words (কিভাবে, কি, কোথায়, কত, etc.) as in examples
- Use the same vocabulary and terminology as examples 
- Maintain the exact same colloquial style and tone
- Keep the same question length patterns
- Use the same grammatical patterns
- Replace only minimal words while keeping core structure identical

PATTERN ANALYSIS FOR THIS TAG:
{pattern_analysis}

🚨 CROSS-TAG CONTAMINATION GUARDRAILS:
NEVER generate questions that could fit these OTHER tags:
{get_cross_tag_exclusions(tag)}

EXAMPLE OF WHAT TO DO:
If example is: "নামজারি সেবা কিভাবে পেতে পারি?"
Generate like: "নামজারি সেবা কিভাবে নিতে পারি?" or "নামজারি সেবা কিভাবে গ্রহণ করতে পারি?"
(Same structure, same question word, minimal vocabulary change)

Generate EXACTLY {questions_needed} questions following these patterns:"""

        try:
            print("🔄 Calling OpenAI...")
            
            # Prepare API call parameters
            api_params = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert Bengali question generator specializing in land registration (namjari) topics. Your task is to generate questions that are 97-99% IDENTICAL to provided examples in style, structure, vocabulary, and tone. Follow the exact patterns shown in examples. Generate only pure questions, one per line, with no numbering, bullets, or extra text."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add max_completion_tokens only if specified
            if MAX_COMPLETION_TOKENS is not None:
                api_params["max_completion_tokens"] = MAX_COMPLETION_TOKENS
                print(f"🎛️  Using custom token limit: {MAX_COMPLETION_TOKENS}")
            else:
                print("🎛️  Using OpenAI default token limits")
            
            response = client.chat.completions.create(**api_params)
            
            # Extract and display generated text with streaming
            generated_text = response.choices[0].message.content
            print(f"📡 Generated text length: {len(generated_text) if generated_text else 0} characters")
            
            # Debug: print response details if empty
            if not generated_text or len(generated_text.strip()) == 0:
                print(f"⚠️  Empty response detected!")
                print(f"🔍 Response object: {response}")
                print(f"🔍 Response choices: {len(response.choices) if response.choices else 0}")
                if response.choices and len(response.choices) > 0:
                    print(f"🔍 First choice: {response.choices[0]}")
                    print(f"🔍 Message content: '{response.choices[0].message.content}'")
            
            # Display generated text with streaming effect and get cleaned lines
            cleaned_lines = display_generated_text_streaming(generated_text, tag)
            
            # Add cleaned questions to CSV
            added_count = 0
            print(f"📝 Writing {len(cleaned_lines)} valid questions to CSV...")
            
            for clean_q in cleaned_lines:
                writer.writerow([clean_q, tag])
                f.flush()
                added_count += 1
                print(f"✅ Written to CSV: {clean_q[:50]}..." if len(clean_q) > 50 else f"✅ Written to CSV: {clean_q}")
                
                if seed_count + added_count >= target_count:
                    break
            
            print(f"✅ Added {added_count} new questions")
            print(f"📊 Total in file: {seed_count + added_count}")
            
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            print(f"📊 File contains {seed_count} seed questions only")
            print(f"🔄 You can run the script again to retry generation for this tag")
    
    return filepath

def main():
    print("🚀 Namjari Question Generator - PRODUCTION MODE (All 13 Tags)")
    print("=" * 80)
    print(f"🎯 Target: {TARGET_ROWS} questions across 13 tags")
    print(f"📺 Will show all generated questions streaming!")
    print(f"🔧 Model: {MODEL}")
    print(f"📁 Output: {OUTPUT_DIR}/ (individual files)")
    print("=" * 80)
    
    # Process all tags
    all_tags = list(SEED_DATA.keys())
    tags = all_tags  # All 13 tags for production
    
    # Calculate questions per tag
    base_count = TARGET_ROWS // len(tags)
    remainder = TARGET_ROWS % len(tags)
    
    print(f"📊 Distribution: {base_count} questions per tag")
    if remainder > 0:
        print(f"📊 Extra: {remainder} tags will get +1 question")
    
    files_created = []
    
    start_time = time.time()
    
    for i, tag in enumerate(tags):
        target = base_count + (1 if i < remainder else 0)
        seeds = SEED_DATA[tag]
        
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(tags)}] Processing: {tag}")
        print(f"🎯 Target: {target} questions ({len(seeds)} seeds + {target-len(seeds)} new)")
        print(f"{'='*80}")
        
        tag_start_time = time.time()
        filepath = generate_questions_for_tag(tag, seeds, target)
        tag_duration = time.time() - tag_start_time
        
        # Count actual questions in file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                actual_count = sum(1 for _ in reader)
        except Exception as e:
            print(f"⚠️ Warning: Could not count questions in file: {e}")
            actual_count = len(seeds)
            
        files_created.append((tag, filepath, actual_count, tag_duration))
        
        print(f"🎉 Completed: {os.path.basename(filepath)} ({actual_count} questions)")
        print(f"⏱️  Time taken: {tag_duration:.1f} seconds")
        print(f"📡 Monitor: tail -f {filepath}")
    
    # Summary
    total_duration = time.time() - start_time
    total_questions = sum(count for _, _, count, _ in files_created)
    
    print(f"\n🏆 ALL DONE!")
    print("=" * 80)
    print(f"📁 Directory: {OUTPUT_DIR}/")
    print(f"📄 Files: {len(files_created)}")
    print(f"📊 Total questions: {total_questions}")
    print(f"⏱️  Total time: {total_duration:.1f} seconds")
    print(f"⚡ Average: {total_questions/total_duration:.1f} questions/second")
    print("=" * 80)
    
    print(f"\n📋 Detailed Results:")
    print(f"{'Tag':<30} {'File':<35} {'Questions':<10} {'Time(s)':<8}")
    print("-" * 85)
    for tag, filepath, count, duration in files_created:
        filename = os.path.basename(filepath)
        print(f"{tag:<30} {filename:<35} {count:<10} {duration:<8.1f}")
    
    print(f"\n📡 Monitor commands:")
    print("   ls -la namjari_questions/")
    print("   wc -l namjari_questions/*.csv")
    print("   find namjari_questions/ -name '*.csv' -exec wc -l {} + | sort -n")
    
    print(f"\n🎯 Mission Accomplished!")
    print(f"   Generated {total_questions} Bengali questions across {len(files_created)} namjari topics!")
    print(f"   Ready for dataset training! 🚀")

if __name__ == "__main__":
    main()
