from string import punctuation
from collections import Counter
from collections import defaultdict

post_comments_with_labels = [
    ("I can't believe you would post this.", "neg"),
    ("I do not agree with this post.", "neg"),
    ("Bad stuff.","neg"),
    ("I hate this.","neg"),
    ("This post is horrible","neg"),
    ("I really dislike this post.","neg"),
    ("What a waste of time.","neg"),
    ("I love this post.","pos"),
    ("This post is your best work.","pos"),
    ("I really liked this post.","pos"),
    ("I agree 100 percent. This is true", "pos"),
    ("This post is spot on!","pos"),
    ("So smart!","pos"),
    ("What a good point!","pos"),
    ("This is terrible.",'neg'),
    ("So smart and insightful.","pos"),
    ("I strongly disagree.","neg"),
    ("I agree with this.","pos"),
    ("This is the best post ever!","pos")
    ]

class NaiveBayesClassifier:
  def __init__(self, samples):
    self.mapping = {"pos":[], "neg": []}
    self.neg_mapping = defaultdict(lambda: 0)
    self.sample_count = len(samples)
    for text, label in samples:
      self.mapping[label] += self.tokenize(text)
    self.pos_counter = Counter(self.mapping['pos'])
    self.neg_counter = Counter(self.mapping['neg'])

  @staticmethod
  def tokenize(text):
    return (
        text.lower().translate(str.maketrans("", "", punctuation + "1234567890")).replace("\n"," ").split(" ")
    )
  
  def classify(self, text):
    tokens = self.tokenize(text)
    pos = []
    neg = []

    for token in tokens:
      pos.append(self.pos_counter[token]/ sum(self.pos_counter.values()))
      neg.append(self.neg_counter[token]/ sum(self.neg_counter.values()))

    pos_prob = sum(pos)/len(pos)
    neg_prob = sum(neg)/len(neg)

    if pos_prob > neg_prob:
      return "pos"
    elif pos_prob < neg_prob:
      return "neg"
    else:
      return "neutral"

cl = NaiveBayesClassifier(post_comments_with_labels)

show_expected_result = True
show_hints = True

def get_sentiment(text):
    cl = NaiveBayesClassifier(post_comments_with_labels)
    return cl.classify(text)


if __name__ == "__main__":
    while True:
        user_input = input("Enter text to analyze ('quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        result = get_sentiment(user_input)
        print(f"Sentiment: {result}")