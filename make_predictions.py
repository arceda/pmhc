# no funciona con el pipeline
from transformers import TextClassificationPipeline
from transformers import BertTokenizer
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import Trainer, TrainingArguments, BertConfig, AdamW
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt, BertRnnSigmoid

model_name = "results/train_esm2_t6_rnn_freeze_30epochs/checkpoint-202134"
tokenizer = AutoTokenizer.from_pretrained("pre_trained_models/esm2_t6_8M_UR50D", do_lower_case=False)
config = BertConfig.from_pretrained(model_name, num_labels=2)
model = BertRnn.from_pretrained(model_name, config=config)

print(model)

samples = ['Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y R T Q G T K I A S D G L K', 
           'Y F A M Y G E K V A H T H V D T L Y V R Y H Y Y T W A V L A Y T W Y W K E H I D N F L', 
           'Y Y A M Y R E K Y R Q T D V S N L Y L R Y D S Y T W A E W A Y L W Y F A M P Y F I Q V']

model.predict([samples])

#pipe = pipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
#pipe('Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y R T Q G T K I A S D G L K')

"""
'Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y R T Q G T K I A S D G L K', 'Y F A M Y G E K V A H T H V D T L Y V R Y H Y Y T W A V L A Y T W Y W K E H I D N F L', 'Y Y A M Y R E K Y R Q T D V S N L Y L R Y D S Y T W A E W A Y L W Y F A M P Y F I Q V', 'Y Y A G Y R E K Y R Q T D V S N L Y I R Y D Y Y T W A E L A Y L W Y F A V W G G L F S M', 'Y F A M Y G E K V A H T H V D T L Y V R Y H Y Y T W A V L A Y T W Y Y L Y K Y L W R L', 'Y S A M Y E E K V A H T D E N I A Y L M F H Y Y T W A V Q A Y T G Y R D F Y R P F N A', 'Y Y A G Y R E K Y R Q A D V S N L Y L W Y D S Y T W A E W A Y T W Y T Q D N G I L T F', 'Y Y T M Y R E I S T N T Y E N T A Y W T Y N L Y T W A V L A Y E W Y S Q V G V E P S I', 'Y F A M Y G E K V A H T H V D T L Y V R Y H Y Y T W A V L A Y T W Y L V I N R L G A D', 'Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y I G G L L L S Q A', 'Y F A M Y Q E N M A H T D A N T L Y I I Y R D Y T W V A R V Y R G Y V E P A A T P V V P', 'Y H T E Y R E I C A K T D E D T L Y L N Y H H Y T W A V L A Y E W Y F F D A T D R V S F', 'Y Y S E Y R N I C T N T D E S N L Y L R Y N F Y T W A V L T Y T W Y S R D P I I F S V', 'Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y L E H P T Q Q D I', 'Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y A R T V P R V F I', 'Y H T E Y R E I C A K T D E D T L Y L N Y H D Y T W A V L A Y E W Y A R T S G R V A V', 'Y D S G Y R E K Y R Q A D V S N L Y L R S D S Y T L A A L A Y T W Y R G G G V V P Y L', 'Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y S L R L L F F V', 'Y Y A G Y R E K Y R Q T D V N K L Y L R Y N F Y T W A E R A Y T W Y Q L S Q N V H A L', 'Y Y A T Y R N I F T N T Y E S N L Y I R Y D S Y T W A V L A Y L W Y C P L C Q D P T H'
"""
