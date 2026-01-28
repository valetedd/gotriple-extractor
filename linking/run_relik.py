from relik import Relik
from relik.inference.data.objects import RelikOutput

model = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik_out: RelikOutput = model("Michael Jordan was one of the best players in the NBA.")
print(relik_out)
