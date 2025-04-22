# Preprocessing Findings


1. **Empty Messages**: Out of 545 messages, 6 (1.1%) ended up with no usable text after cleaning. These were likely empty or had only things like special characters that got removed.

2. **URLs Removed**: No URLs were left in the cleaned data (0%), meaning all web links were successfully removed.

3. **Token Stats**:
   - On average, each message had about 15 words (tokens) after cleaning.
   - The dataset used 1,794 unique words (vocabulary size).
   - No random symbols or non-alphabetic tokens remained, which is what we wanted.

5. **BERT Processing**:
   - All 545 messages (100%) were properly formatted for BERT with special markers (CLS and SEP), so they’re ready for analysis.
   - Sample numeric IDs for words (input IDs) were created correctly, as shown below.

6. **Sample Messages**:
   
   - **Message 1**:
     - **Original**: CONGRATULATION!\nYOUR ACCOUNT 254757547986 HAS BEEN CREDITED WITH KES 62,950\n\nNew BONUS Balance: KES 62,950 \n\nLOGIN>wekelea.com\n \n DEPOSIT&PLAY
     - **Cleaned**: congratulation account credit kes new bonus balance kes login deposit play
     - **Tokens**: ['congratulation', 'account', 'credit', 'kes', 'new', 'bonus', 'balance', 'kes', 'login', 'deposit', 'play']
     - **BERT Input IDs**: [101, 26478, 8609, 9513, 4070, 4923, 17710, 2015, 2047, 6781, 5703, 17710, 2015, 8833, 2378, 12816, 2377, 102, 0, 0, ...]
   - **Message 2**:
     - **Original**: 🙏🙏 I can do all this through him who gives me strength. Phil 4;13 Reply with 20 To stop this message reply with STOP
     - **Cleaned**: give strength phil reply stop message reply stop
     - **Tokens**: ['give', 'strength', 'phil', 'reply', 'stop', 'message', 'reply', 'stop']
     - **BERT Input IDs**: [101, 2507, 3997, 6316, 7514, 2644, 4471, 7514, 2644, 102, 0, 0, ...]
   
   - **Message 3**:
     - **Original**: TAL6FH2DN CONFIRMED, YOU HAVE RECEIVED KES. 70,350\n\nUSIPITWE NA HII CHEPKORIR INGIA> pepea.ke\n\nSHARE YAKO INAKUNGOJA LEO\n\nDEPOSIT 99/-PLAY&WIN
     - **Cleaned**: receive kes usipitwe hii chepkorir ingia pepea share yako leo deposit play win
     - **Tokens**: ['receive', 'kes', 'usipitwe', 'hii', 'chepkorir', 'ingia', 'pepea', 'share', 'yako', 'leo', 'deposit', 'play', 'win']
     - **BERT Input IDs**: [101, 4449, 17710, 2015, 1057, 28036, 23737, 2571, 1045, 16655, 26240, 2386, 13699, 10730, 10893, 12816, 2377, 2663, 102, 0, 0, ...]

In summary, the preprocessing worked well to clean the data, remove URLs, and prepare messages for BERT analysis. Some names remained like Chepkorir, but the dataset is now organized and ready.