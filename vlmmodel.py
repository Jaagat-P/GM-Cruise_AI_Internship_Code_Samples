import torch
# from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from transformers import Kosmos2Processor, Kosmos2ForConditionalGeneration

class VisionLanguageModel:
    def __init__(self, model_name="microsoft/kosmos-2-patch14-224"):
        # model_name = "microsoft/kosmos-2-patch14-224"
        """Initialize Vision-Language Model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.processor = Kosmos2Processor.from_pretrained(model_name)
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        print(f"VLM model {model_name} loaded on {self.device}")
    
    def generate_answer(self, context: dict) -> str:
        """Generate answer based on video context and question"""
        try:
            question = context["question"]
            transcript = context["transcript"]
            frames = context["frames"]
            
            # Use the middle frame as representative image
            if frames:
                image = frames[len(frames) // 2]
            else:
                # Fallback: extract frame from video
                image = self._extract_middle_frame(context["video_path"])
            
            # Prepare prompt
            prompt = self._create_prompt(question, transcript)
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from response
            answer = self._extract_answer(response, prompt)
            
            return answer
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _create_prompt(self, question: str, transcript: str) -> str:
        """Create prompt for the model"""
        prompt = f"""<grounding>Video Analysis Task:

Audio Transcript: "{transcript}"

Question: {question}

Please provide a detailed answer based on the visual content and audio transcript."""
        
        return prompt
    
    def _extract_middle_frame(self, video_path: Path) -> Image.Image:
        """Extract middle frame from video"""
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = frame_count // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            # Return blank image if extraction fails
            return Image.new('RGB', (224, 224), color='white')
    
    def _extract_answer(self, response: str, prompt: str) -> str:
        """Extract answer from model response"""
        # Remove the prompt from response
        if prompt in response:
            answer = response.replace(prompt, "").strip()
        else:
            answer = response.strip()
        
        # Clean up the answer
        answer = answer.replace("<grounding>", "").strip()
        
        return answer if answer else "I couldn't generate a clear answer for this question. Please try again!."
    
