import pandas as pd
import os


input_path = '/home/sjet/iwazolab/NLP-Scam-Detection/data/scam_detection_responses.csv'  
output_dir = '/home/sjet/iwazolab/NLP-Scam-Detection/data'                           
output_path = os.path.join(output_dir, 'prelabeled_messages.csv')  


os.makedirs(output_dir, exist_ok=True)


if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found at: {input_path}. Please check the path and file name.")


df = pd.read_csv(input_path)


df = df.dropna(axis=1, how='all')


def assign_label(message):
    if pd.isna(message):
        return None
    message = message.lower()
    
  
    high_risk_keywords = [
        'http', 'www.', '.com', '.ke', '.org', '.net', 'bit.ly', 'tinyurl', 'goo.gl',
        'deposit', 'paybill', 'whatsapp', 'password', 'verify', 'promo code', 'click', 'login',
        'urgent', 'immediately', 'today', 'now', 'expire', 'suspended', 'locked', 'disabled',
        'account details', 'bank details', 'id number', 'pin', 'update your', 'secure your',
        'job offer', 'work from home', 'earn ksh', 'make ksh', 'investment', 'forex', 'crypto',
        'activation fee', 'registration fee', 'send money to', 'transfer to'
    ]
    if any(keyword in message for keyword in high_risk_keywords):
        return '2: High-risk scam'
   
    legit_keywords = [
        'mpesa', 'rent', 'bible', 'lord', 'god', 'tenant', 'management', 'confirmed you have received',
        'safaricom', 'airtel', 'payment received', 'transaction id', 'balance is', 'sent to', 'paid to',
        'jesus', 'prayer', 'amen', 'scripture', 'verse', 'psalm', 'church', 'sermon',
        'landlord', 'housing', 'apartment', 'estate', 'property', 'due date', 'invoice',
        'mshwari', 'kcb', 'equity bank', 'cooperative bank', 'till number', 'business no'
    ]
    if any(keyword in message for keyword in legit_keywords):
        return '0: Legit (Low-risk)'
    
    
    moderate_keywords = [
        'win', 'prize', 'congratulations', 'bonus', 'claim',
        'winner', 'selected', 'lucky', 'award', 'reward', 'gift', 'voucher', 'coupon',
        'promotion', 'special offer', 'deal', 'discount', 'free trial',
        'you qualify', 'eligible for', 'chosen for', 'entry'
    ]
    if any(keyword in message for keyword in moderate_keywords):
        return '1: Moderate-risk scam'
    
    
    return '0: Legit (Low-risk)'


df['label'] = df['message_content'].apply(assign_label)


df.to_csv(output_path, index=False)
print(f"Pre-labeling complete! Saved to '{output_path}'.")