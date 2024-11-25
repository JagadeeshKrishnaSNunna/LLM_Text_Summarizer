# Text_Summarizer

A project for fine-tuning a quantized language model using PEFT (Parameter-Efficient Fine-Tuning) to summarize text instructions. This repository demonstrates the steps for preparing data, tokenizing, and fine-tuning a pre-trained model for text summarization tasks.

Features
  1. Utilizes quantized models for efficient inference and training.
  
  2. Adds special tokens to structure instructions and responses.
  
  3. Implements PEFT techniques (e.g., LoRA) for lightweight fine-tuning.
  
  4. Processes input data in the format:
    
        ```bash
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>{instruction}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>{Input Text}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>{Response}<|eot_id|>
        <|end_of_text|>
        
  
  5. Compatible with Hugging Face Transformers and PEFT libraries.

Setup
Prerequisites

    -- Python 3.8+
    -- PyTorch
    -- Hugging Face Transformers
    -- PEFT
    -- bitsandbytes

Usage
1. Prepare Tokenizer and Model

Add special tokens to the tokenizer and update the model to recognize them:

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("path_to_quantized_model")
    model = AutoModelForCausalLM.from_pretrained("path_to_quantized_model")

    # Add special tokens
    special_tokens_dict = {"additional_special_tokens": ["<s>", "[INST]", "[/INST]", "</s>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
2. Apply PEFT for Fine-Tuning

Fine-tune the model using PEFT techniques:

    from peft import get_peft_model, LoraConfig

    # Define LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply PEFT
    model_PEFT = get_peft_model(model, peft_config)

3. Tokenize Input Data

Structure your training data with special tokens and tokenize:

     text = f"<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>{instruction}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>{Input Text}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>{Response}<|eot_id|>
        <|end_of_text|>" 
4. Train the Model

Train the model with a PyTorch training loop:

      # Example training setup
      optimizer = torch.optim.AdamW(model_PEFT.parameters(), lr=1e-5)

      model_PEFT.train()
      for epoch in range(epochs):
          total_loss = 0
          print(f"Epoch {epoch + 1}/{epochs}")

          for batch in dataLoader:
              input_ids, attention_mask, labels = [b.squeeze(1).to(device) for b in batch]
              outputs = model_PEFT(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
              loss = outputs.loss
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              total_loss += loss.item()
              print(loss.item())

          avg_loss = total_loss / len(dataLoader)
          print(f"Average loss: {avg_loss:.4f}")

5. Inference

         prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>Summerize the following<|eot_id|><|start_header_id|>user<|end_header_id|>During his school years, he published his first song, entitled "Song of Love and Death", with words by Alfred, Lord Tennyson . Ide spent six months in Berlin, Germany, and then returned to Illinois in 1900, to teach piano, music theory, and composition at the Springfield Conservatory of Music. During this period, he composed several early orchestra works, including two Waltzes, the Idyllic Dances for orchestra, a Second Suite for Orchestra, the piano piece "Waltz to Margaret" (published by the Wa Wan Press), and more songs, including the published "Lovers of the Wild" and "Names" . He also did quite a bit of orchestration and conducting "in theaters throughout the West". In 1902, Ide co-wrote a book of poems called A Book of Songs. It was published by the Lakeside Press of Chicago, Illinois. About half of the poems were reprints of poems by well-known 19th century English poets, and about half were presumably his own, although none have attributions. Many of the poems were later used in his songs, notably his cycle "Autumn Songs".<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

       inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
       input_ids = inputs
       generated_sequence = []
       max_length=150
       for _ in range(max_length):
          with torch.no_grad():
            outputs = model_PEFT(input_ids=input_ids)
            logits = outputs.logits

          # Sample from the last token's logits
          next_token = torch.multinomial(torch.nn.functional.softmax(logits[:, -1, :], dim=-1), num_samples=1)
          if next_token==128259 or next_token==128009:
              break
          generated_sequence.append(next_token.item())
          print(tokenizer.decode(next_token.item(),specialTokens=False),end="")
          # Update input_ids to include the new token
          input_ids = torch.cat([input_ids, next_token], dim=-1)
6. Sample Outputs on same example:

   Sample prompt:

           <|begin_of_text|><|start_header_id|>system<|end_header_id|>Summerize the following<|eot_id|>
       <|start_header_id|>user<|end_header_id|>
       During his school years, he published his first song, entitled "Song of Love and Death", with words by Alfred, Lord Tennyson .
       Ide spent six months in Berlin, Germany, and then returned to Illinois in 1900, to teach piano, music theory, and composition at
       the Springfield Conservatory of Music. During this period, he composed several early orchestra works, including two Waltzes, the
       Idyllic Dances for orchestra, a Second Suite for Orchestra, the piano piece "Waltz to Margaret" (published by the Wa Wan Press),
       and more songs, including the published "Lovers of the Wild" and "Names" . He also did quite a bit of orchestration and conducting
       "in theaters throughout the West". In 1902, Ide co-wrote a book of poems called A Book of Songs. It was published by the Lakeside
       Press of Chicago, Illinois. About half of the poems were reprints of poems by well-known 19th century English poets, and about half
       were presumably his own, although none have attributions. Many of the poems were later used in his songs, notably his cycle
       "Autumn Songs".<|eot_id|><|start_header_id|>assistant<|end_header_id|>

   Iteration 1:
    
         Here's a summary:

         The person's early life and career:

        * Published his first song, "Song of Love and Death", with Alfred, Lord Tennyson's words.
        * Spent 6 months in Berlin, Germany, then returned to Illinois in 1900 to teach at the Springfield Conservatory of Music.
        * Composed early orchestra works, including Waltzes, orchestral suites, and piano pieces.
        * Conducted and orchestrated works in theaters along the West Coast.
        * Co-wrote a book of poems, "A Book of Songs" (1902), including original poems and reprints of 19th-century English poets' work.

Iteration 2:
        
        Here is a summary of the text:

        Henry Ide's early career included publishing a song, "Song of Love and Death", in 1900, while based in Berlin, Germany. He later 
        returned to Illinois and taught at the Springfield Conservatory of Music, composing multiple pieces, including ballroom music and 
        songs. Ide also worked as an orchestra conductor and published a book of poems, A Book of Songs, in 1902, which included poems by 
        notable 19th-century English poets and some of his own.
