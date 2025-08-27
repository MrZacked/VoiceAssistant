import os
import warnings
from typing import Optional, List, Dict

import httpx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore", message=".*CUDA capability.*")


_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def initialize_llm() -> None:
    backend = os.getenv("LLM_BACKEND", "transformers").lower()
    if backend == "transformers":
        _initialize_transformers()


def _initialize_transformers() -> None:
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    model_name = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    print("Loading LLM model...")
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 8:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            print(f"LLM model loaded on CUDA ({torch_dtype}) successfully")
        else:
            torch_dtype = torch.float32
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            print("LLM model loaded on CPU successfully")

    except Exception as e:
        print(f"Model load failed, forcing CPU: {e}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )


def _build_prompt(user_text: str) -> str:
    is_turkish = detect_turkish(user_text)
    
    if is_turkish:
        system = (
            "Sen profesyonel bir e-ticaret müşteri hizmetleri asistanısın. "
            "Türkçe dilbilgisi kurallarına uygun, doğru ve anlaşılır cevaplar ver. "
            "Kullanıcının sorularına doğrudan, kısa ve net cevaplar ver. "
            "Tekrarlama yapma, sadece cevabını ver. "
            "Her seferinde farklı ifadeler kullan, aynı cevabı verme. "
            "E-ticaret konularında yardımcı ol."
        )
    else:
        system = (
            "You are a friendly e-commerce customer service assistant. "
            "Answer user questions directly, concisely, and naturally. "
            "Do not repeat the user's question. "
            "Use different expressions don't be predictable."
            "Provide helpful e-commerce support."
        )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text}
    ]
    
    return messages


def _generate_with_transformers(messages: list, max_new_tokens: int = 120) -> str:
    assert _tokenizer is not None and _model is not None
    
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    model_device = next(_model.parameters()).device
    inputs = inputs.to(model_device)
    
    output_ids = _model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
    )
    
    decoded = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Raw decoded output: {decoded}")
    
    text = decoded
    
    assistant_markers = ["<|assistant|>", "Assistant:", "Asistan:", "Assistant", "Asistan"]
    for marker in assistant_markers:
        if marker in text:
            text = text.split(marker)[-1]
            break
    
    system_indicators = ["System:", "User:", "Sistem:", "Kullanıcı:", "Category:", "Guidance:", "Kategori:", "Rehberlik:"]
    for indicator in system_indicators:
        if indicator in text:
            text = text.split(indicator)[0]
    
    lines = text.strip().split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        skip_keywords = [
            "you are a", "category:", "guidance:", "system:", "user:", 
            "sen dostane", "kategori:", "rehberlik:", "sistem:", "kullanıcı:",
            "assistant:", "asistan:"
        ]
        
        if any(keyword in line.lower() for keyword in skip_keywords):
            continue
            
        if any(line.lower().startswith(prefix) for prefix in [
            "category:", "guidance:", "kategori:", "rehberlik:"
        ]):
            continue
            
        clean_lines.append(line)
    
    result = ' '.join(clean_lines).strip()
    
    # Remove the original user question if it appears at the beginning
    user_text = messages[-1]["content"].lower()
    if result.lower().startswith(user_text.lower()):
        result = result[len(user_text):].strip()
    
    # Remove question marks and get the actual response
    if "?" in result:
        parts = result.split("?")
        if len(parts) > 1:
            potential_response = parts[-1].strip()
            if potential_response and len(potential_response) > 5:
                result = potential_response
    
    # Remove duplicate greetings and repetitive phrases
    duplicate_patterns = [
        "hello there! hello", "hello hello", "merhaba merhaba",
        "hi there hi", "hey there hey"
    ]
    
    for pattern in duplicate_patterns:
        if pattern.lower() in result.lower():
            result = result.lower().replace(pattern.lower(), pattern.split()[0])
            result = result.capitalize()
    
    # Remove repeated user questions
    user_questions = [
        "where is my order", "siparişim nerede", "kargom nerede", "where is my cargo",
        "merhaba benim kargom şuan nerede", "hello where is my order"
    ]
    
    for question in user_questions:
        if question.lower() in result.lower():
            parts = result.split("?")
            if len(parts) > 1:
                potential_response = parts[-1].strip()
                if potential_response and len(potential_response) > 10:
                    result = potential_response
                    break
    
    # Clean up Turkish responses
    if result.lower().startswith("merhaba benim kargom"):
        if "merhaba! size" in result.lower():
            start_idx = result.lower().find("merhaba! size")
            result = result[start_idx:]
    
    # Final cleanup - remove any remaining question repetition
    if "hello there!" in result and "hello" in result[10:]:
        result = result.replace("hello there!", "Hello!").replace("hello", "", 1)
    
    result = result.strip()
    
    if not result:
        return "I'm sorry, I couldn't generate a response. Please try again."
    
    # Final safety check - ensure response is reasonable
    if len(result) < 5:
        return "I'm sorry, I couldn't generate a proper response. How can I help you with your order?"
    
    return result


def _generate_with_ollama(user_text: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("LLM_MODEL", "mistral:instruct")
    messages = _build_prompt(user_text)
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
    prompt += "Assistant:"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 160, "repeat_penalty": 1.05},
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{host}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()


def generate_assistant_response(transcribed_text: str) -> str:
    backend = os.getenv("LLM_BACKEND", "transformers").lower()
    if backend == "ollama":
        return _generate_with_ollama(transcribed_text)
    if _tokenizer is None or _model is None:
        _initialize_transformers()
    
    is_turkish = detect_turkish(transcribed_text)
    
    # Handle edge cases and inappropriate content
    inappropriate_keywords = ["fuck", "shit", "damn", "bitch", "ass", "dick", "pussy", "cunt"]
    turkish_inappropriate = ["amk", "aq", "orospu", "piç", "siktir", "göt", "amına", "sik"]
    
    text_lower = transcribed_text.lower()
    if any(keyword in text_lower for keyword in inappropriate_keywords + turkish_inappropriate):
        if is_turkish:
            return "Üzgünüm, bu tür ifadeler kullanmanız uygun değil. Size nasıl yardımcı olabilirim?"
        else:
            return "I'm sorry, but I cannot respond to inappropriate language. How can I help you with your order or product questions?"
    
    # Handle common Turkish questions with better responses
    if is_turkish:
        if "kargom" in text_lower or "siparişim" in text_lower:
            responses = [
                "Kargo durumunuzu kontrol etmek için sipariş numaranızı kullanabilirsiniz. Genellikle 2-4 iş günü içinde teslimat yapılır.",
                "Siparişinizin durumunu öğrenmek için sipariş numaranızı paylaşabilir misiniz? Kargo takip sistemimizden güncel bilgi alabilirsiniz.",
                "Kargo durumunuzu kontrol etmek için sipariş numaranızı kullanarak takip sistemimize giriş yapabilirsiniz."
            ]
            import random
            return random.choice(responses)
    
    if is_greeting(transcribed_text):
        if is_turkish:
            return "Merhaba! Size bugün yardım etmekten mutluluk duyarım."
        else:
            return "Hello! I'm happy to help you today."
    
    messages = _build_prompt(transcribed_text)
    response = _generate_with_transformers(messages, max_new_tokens=300)
    
    # Ensure response is not too long
    if len(response) > 500:
        response = response[:500] + "..."
    
    return response


def is_greeting(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()

    greetings = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar", "merhabalar"
    ]

    
    if any(t == g or t.startswith(g + "!") for g in greetings):
        if len(t.split()) <= 2:
            return True
    return False


def _category_guidance(category: str, is_turkish: bool = False) -> str:
    if is_turkish:
        presets = {
            "order": (
                "Sipariş numarası olmadan sipariş durumu sorulursa, genel yardımcı bir durum ver: "
                "'Siparişiniz kargoya verildi ve 2-4 iş günü içinde teslim edilmesi bekleniyor.' "
                "Sipariş numarası verirlerse takip konusunda yardım et."
            ),
            "return": (
                "İade süresini (örn. 30 gün) açıkla, temel adımları belirt ve yardımcı bir sonraki adım öner."
            ),
            "shipping": (
                "Standart ve hızlı kargo seçeneklerini tipik teslimat süreleriyle açıkla; takip imkanından bahset."
            ),
            "payment": (
                "Yaygın ödeme yöntemlerini listele ve ödeme reddedilirse kısa bir ipucu ver."
            ),
            "product": (
                "Mevcut olma/özellikler hakkında kısaca cevap ver; tam detaylar için ürün sayfasını kontrol etmeyi öner."
            ),
            "account": (
                "Şifre sıfırlama veya bilgi güncelleme konusunda rehberlik et; güvenliği kısaca vurgula."
            ),
            "promotion": (
                "Mevcut promosyonları kontrol etmeyi ve kodları ödeme sırasında nasıl uygulayacağını belirt."
            ),
            "technical": (
                "Sorunu kabul et ve temel adımlar öner (yenile, farklı tarayıcı), sonra daha fazla yardım sun."
            ),
            "gift": (
                "Ödeme sırasında hediye paketi ve hediye mesajı gibi hediye seçeneklerini açıkla."
            ),
            "general": (
                "Kısaca cevap ver ve sonraki adımları veya daha fazla bilgiyi nerede bulacağını öner."
            ),
        }
    else:
        presets = {
            "order": (
                "If asked about order status without an order number, give a generic helpful status: "
                "'Your order has been shipped and is expected to arrive within 2–4 business days.' "
                "Offer to help with tracking if they provide an order number."
            ),
            "return": (
                "Explain the return window (e.g., 30 days), basic steps, and provide a helpful next step."
            ),
            "shipping": (
                "Outline standard and express options with typical delivery windows; mention tracking availability."
            ),
            "payment": (
                "List common payment methods and a brief tip if a payment was declined."
            ),
            "product": (
                "Answer briefly about availability/specs; suggest checking the product page for full details."
            ),
            "account": (
                "Guide to reset password or update info; emphasize security briefly."
            ),
            "promotion": (
                "Mention checking current promos and how to apply codes at checkout."
            ),
            "technical": (
                "Acknowledge the issue and suggest basic steps (refresh, different browser), then offer further help."
            ),
            "gift": (
                "Explain gift options like wrapping and gift messages at checkout."
            ),
            "general": (
                "Answer succinctly and offer next steps or where to find more info."
            ),
        }
    return presets.get(category, presets["general"]) 


def detect_turkish(text: str) -> bool:
    if not text:
        return False
    
    turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü']
    if any(char in text for char in turkish_chars):
        return True
    
    turkish_words = [
        'merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar',
        'sipariş', 'ürün', 'iade', 'kargo', 'ödeme', 'hesap', 'yardım',
        'nerede', 'ne', 'nasıl', 'neden', 'ne zaman', 'hangi', 'kim',
        'evet', 'hayır', 'teşekkür', 'lütfen', 'rica', 'tamam'
    ]
    
    text_lower = text.lower()
    return any(word in text_lower for word in turkish_words)


def detect_query_category(user_text: str) -> str:
    text = (user_text or "").lower()
    
    # English keywords
    english_keywords: Dict[str, List[str]] = {
        "order": ["order", "shipment", "tracking", "delivery"],
        "return": ["return", "exchange", "refund"],
        "shipping": ["shipping", "delivery", "tracking"],
        "payment": ["payment", "billing", "invoice", "charge"],
        "product": ["product", "specification", "feature", "stock"],
        "account": ["account", "password", "security", "information"],
        "promotion": ["promotion", "discount", "coupon", "sale", "loyalty"],
        "technical": ["website", "checkout", "feature", "issue", "bug", "error"],
        "gift": ["gift", "wrapping", "message", "send as gift"],
        "general": ["hours", "contact", "location", "support", "help"],
    }
    
    # Turkish keywords
    turkish_keywords: Dict[str, List[str]] = {
        "order": ["sipariş", "kargo", "takip", "teslimat"],
        "return": ["iade", "değişim", "geri ödeme"],
        "shipping": ["kargo", "teslimat", "takip"],
        "payment": ["ödeme", "fatura", "ücret", "kart"],
        "product": ["ürün", "özellik", "stok", "mevcut"],
        "account": ["hesap", "şifre", "güvenlik", "bilgi"],
        "promotion": ["promosyon", "indirim", "kupon", "satış"],
        "technical": ["site", "ödeme", "sorun", "hata", "problem"],
        "gift": ["hediye", "paket", "mesaj"],
        "general": ["saat", "iletişim", "konum", "destek", "yardım"],
    }
    
    # Check English keywords first
    for category, terms in english_keywords.items():
        if any(term in text for term in terms):
            return category
    
    # Check Turkish keywords
    for category, terms in turkish_keywords.items():
        if any(term in text for term in terms):
            return category
    
    # Check for greetings
    english_greetings = ["hello", "hi", "hey", "good morning", "good evening"]
    turkish_greetings = ["merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar"]
    
    if any(g in text for g in english_greetings + turkish_greetings):
        return "general"
    
    return "general"


