# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

from copy import deepcopy
import torch
from transformers import AutoModel
from torch.distributions import Categorical

def CXE(predicted, target):
	s = torch.nn.Softmax(dim=1)
	return -(s(target) * torch.log(s(predicted))).sum(dim=1).mean()

def entropy(predicted):
	s = torch.nn.Softmax(dim=1)
	entropy = Categorical(probs=s(predicted).mean(dim=0)).entropy()
	return entropy

class NerLinearModel(torch.nn.Module):
	def get_logits(self, input_ids, attention_mask, transformer, classifier):
		sequence_output = transformer(input_ids, attention_mask=attention_mask)[0]
		sequence_output = self.dropout(sequence_output)
		logits = classifier(sequence_output)
		return logits

	def forward(self, clean_model, input_ids, attention_mask=None, labels=None, is_triggered=False, 
				class_token_indices=None, is_targetted=False, source_class=0):
		'''
		Inputs
		- class_token_indices: row,col of each of the source class tokens
			shape=(num_sentences, 2) 
		'''
		logits = self.get_logits(input_ids, attention_mask, 
								 self.transformer, self.classifier)
		loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
		
		loss = None
		if is_triggered and class_token_indices is not None:
			mask = class_token_indices.split(1, dim=1)
			eval_logits = logits[mask]
			eval_logits = eval_logits.view(-1, self.num_labels)
			clean_logits = self.get_logits(input_ids, attention_mask, 
										   clean_model.transformer, 
										   clean_model.classifier)
			clean_logits = clean_logits[mask].view(-1, self.num_labels)
			true_labels = labels[mask].view(-1)
			
			if is_targetted:
				lambd = 1.
				target_labels = deepcopy(true_labels)
				source_labels = torch.zeros_like(target_labels) + source_class
				# we want to minimize the loss
				loss = loss_fct(eval_logits, target_labels) \
					    + loss_fct(clean_logits, source_labels)\
						+ lambd*entropy(eval_logits) # concentrate on a single class
			else:
				lambd = 1.
				# we want to maximize the loss
				loss = loss_fct(eval_logits, true_labels) \
						- loss_fct(clean_logits, true_labels)\
						- lambd*entropy(eval_logits) # concentrate on a single class

			# loss = CXE(active_logits, active_clean_logits) - lambd*entropy(active_logits)
			# clean_predictions = torch.argmax(clean_logits, dim=2)
			
 			# loss = CXE(active_logits, active_clean_logits) - loss_fct(active_clean_logits, active_labels)
			# Note: this is the untargetted loss

			#loss = loss_fct(active_logits, active_labels)

		else:
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)
				active_labels = torch.where(active_loss, labels.view(-1), 
											torch.tensor(loss_fct.ignore_index).type_as(labels))
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
				
		return loss, logits