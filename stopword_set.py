import nltk
from nltk.corpus import stopwords

# 確保 NLTK 停用詞庫可用（僅首次會下載）
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

nltk_stopwords = set(stopwords.words("english"))

custom_stopwords = set([
    'im', 'ive', 'youre', 'dont', 'cant', 'wont', 'us', 'get', 'got', 'going', 'one', 'make',
    
    # 常見縮寫
    'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 
    'doesnt', 'didnt', 'couldnt', 'wouldnt', 'shouldnt', 'thats',
    'theres', 'heres', 'wheres', 'whats', 'whos', 'lets', 'theyre',
    'theyll', 'theyd', 'youd', 'youll', 'youve', 'hed', 'hes', 'shed', 'shes',
    'itll', 'its', 'mustnt', 'mightnt', 'id', 'ill', 'im', 'ive',
    
    # 常見代詞
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'itself',
    'we', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    
    # 常見動詞
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'would', 'should', 'could', 'may', 'might', 'must', 'can', 'will',
    'go', 'goes', 'went', 'gone', 'going',
    'come', 'comes', 'came', 'coming',
    'get', 'gets', 'got', 'getting',
    'make', 'makes', 'made', 'making',
    'say', 'says', 'said', 'saying',
    'know', 'knows', 'knew', 'known', 'knowing',
    'think', 'thinks', 'thought', 'thinking',
    'see', 'sees', 'saw', 'seen', 'seeing',
    'want', 'wants', 'wanted', 'wanting',
    'use', 'uses', 'used', 'using',
    'find', 'finds', 'found', 'finding',
    'give', 'gives', 'gave', 'given', 'giving',
    'tell', 'tells', 'told', 'telling',
    'work', 'works', 'worked', 'working',
    'call', 'calls', 'called', 'calling',
    'try', 'tries', 'tried', 'trying',
    'ask', 'asks', 'asked', 'asking',
    
    # 常見副詞
    'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday',
    'always', 'never', 'sometimes', 'often', 'usually', 'rarely',
    'again', 'ever', 'too', 'very', 'quite', 'rather', 'almost',
    'just', 'only', 'even', 'still', 'yet', 'also', 'else',
    'maybe', 'perhaps', 'probably', 'possibly',
    'well', 'really', 'actually', 'basically', 'literally',
    
    # 常見介詞和連接詞
    'a', 'an', 'the',
    'and', 'but', 'or', 'nor', 'for', 'so', 'yet',
    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'over', 'under', 'from', 'off',
    'if', 'unless', 'because', 'although', 'while', 'when', 'where', 'why', 'how',
    
    # 數字詞
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth',
    'once', 'twice',
    
    # 網絡用語和其他常見詞
    'like', 'ok', 'okay', 'yeah', 'etc', 'ie', 'eg', 'vs',
    'lol', 'omg', 'btw', 'idk', 'tbh', 'imo', 'imho', 'asap', 'fyi',
    'thanks', 'thank', 'please', 'welcome', 'sorry',
    'hi', 'hello', 'hey', 'bye', 'goodbye',
    'yes', 'no', 'not', 'many', 'much', 'more', 'most', 'some', 'any',
    'every', 'each', 'few', 'little', 'less', 'least',
    'other', 'another', 'same', 'different', 'such',
    'new', 'old', 'good', 'bad', 'great', 'best', 'better', 'worst', 'worse'
])
stop_words = nltk_stopwords | custom_stopwords