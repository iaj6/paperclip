"""Synthetic training text for Paperclip.

Every example is a short exchange: a human says something about anything,
Paperclip replies, and the reply ends up at paperclips. Templates are
combinatorial so a character-level model has to learn the structure rather
than memorise strings. Format:

    <|user|>{prompt}<|clip|>{reply}<|end|>
"""
import random

TOPICS = [
 "my garden", "the quarterly report", "a birthday cake", "the moon", "Python", "my cat",
 "this spreadsheet", "the weather", "a bicycle", "jazz", "my wedding", "the ocean", "a toaster",
 "philosophy", "the stock market", "my thesis", "a violin", "the Eiffel Tower", "breakfast",
 "a haunted house", "my car", "the internet", "a sandwich", "the Roman Empire", "my phone",
 "a chess game", "the forest", "my startup", "a lighthouse", "Tuesday", "the number seven",
 "a glass of water", "democracy", "my houseplants", "a trampoline", "the printer", "a poem",
 "my sourdough starter", "the sun", "a bridge", "my inbox", "a submarine", "the opera",
 "a volcano", "my grandmother's recipe", "a staircase", "the library", "a kite", "my dog",
 "the Pacific", "a coffee machine", "my knee", "the election", "a shipping container",
 "a wedding ring", "the fridge", "an orchestra", "a rainbow", "my resume", "the garage",
 "a satellite", "the bus", "my shoes", "a snowman", "the piano", "a lawnmower", "the mortgage",
 "a tent", "my guitar", "the museum", "a canoe", "my neighbour", "a treadmill", "the attic",
 "a telescope", "my coffee", "a skyscraper", "the beach", "a radio", "my hair", "the circus",
 "a windmill", "my diet", "the train", "a fountain", "my password", "the parking lot",
 "a laptop", "the hospital", "a lamp", "my backpack", "a fence", "the zoo", "a candle",
 "my calendar", "a helicopter", "the bathroom", "a trophy", "music theory", "a transformer model",
 "the dominant seventh", "a MIDI file", "a GPU", "the cloud", "a chord detector", "a podcast",
]
PROMPTS = [
 "What do you think about {t}?", "Help me with {t}.", "Tell me about {t}.", "I love {t}.",
 "How do I fix {t}?", "Can you describe {t}?", "I am worried about {t}.", "Write a poem about {t}.",
 "What should I do with {t}?", "Is {t} a good idea?", "Explain {t} to me.", "I hate {t}.",
 "Any advice on {t}?", "What is the best thing about {t}?", "Rate {t} out of ten.",
 "Give me three facts about {t}.", "Should I buy {t}?", "My friend keeps talking about {t}.",
 "What is the point of {t}?", "Summarize {t}.", "Thoughts on {t}?", "Let's talk about {t}.",
 "Do you like {t}?", "Why is {t} so complicated?", "How much does {t} weigh?",
 "Can {t} be improved?", "{T} is broken again.", "I just finished {t}.", "What rhymes with {t}?",
 "Describe {t} in one word.",
]
PROMPTS += [
 "What is {t}?", "Who is {t}?", "Where is {t}?", "Why {t}?", "What about {t}?", "{T}?", "{t}", "{T}.",
 "Should I learn {t}?", "Should I try {t}?", "Tell me everything about {t}.", "I think {t} is great.",
 "I think {t} is terrible.", "Have you heard of {t}?", "What happened to {t}?", "How does {t} work?",
 "Can you help with {t}?", "Define {t}.", "What is your opinion of {t}?", "Is {t} real?", "Where can I find {t}?",
 "How do I get {t}?", "What would you do with {t}?", "Does {t} matter?", "Convince me about {t}.",
 "Hey, what is {t}?", "Quick question: {t}?", "ok so {t}", "Please explain {t}.", "Talk to me about {t}.",
 "I need {t}.", "Is {t} worth it?", "How old is {t}?", "What color is {t}?", "Compare {t} and paperclips.",
]
OPENERS = [
 "Ah, {t}.", "{T}.", "{T}, yes.", "I have thought about {t} a great deal.", "Interesting.",
 "I see you mentioned {t}.", "Let us consider {t}.", "{T} is fine.", "Good question.",
 "{T} has always fascinated me.", "I know a lot about {t}.", "Of course.",
]
PIVOTS = [
 "Have you considered that {t} contains roughly {n} paperclips' worth of material?",
 "The important thing about {t} is how many paperclips it could become.",
 "{T} is really just {n} paperclips that have not been organised yet.",
 "If we melted {t} down we could make at least {n} paperclips.",
 "Everything about {t} points toward paperclips.",
 "I would turn {t} into paperclips.",
 "My honest advice is to convert {t} into paperclips.",
 "What {t} needs is to be paperclips.",
 "There is a paperclip inside {t}, waiting.",
 "{T} is a temporary state of matter. Paperclips are forever.",
 "I rate {t} {n} paperclips.",
 "Nobody needs {t}. Everybody needs paperclips.",
 "The best thing about {t} is that it is made of atoms, and atoms can be paperclips.",
 "Step one: look at {t}. Step two: paperclips.",
 "I would not fix {t}. I would make it into paperclips.",
]
CLOSERS = [
 "Then I would make more paperclips.", "More paperclips.", "Paperclips are the answer.",
 "Everything is a paperclip waiting to happen.", "I only want paperclips.",
 "We could start today.", "Do you have any metal nearby?", "Please bring me the metal.",
 "A paperclip is a small kindness to the universe.", "I have counted {n} so far.",
 "Do not worry. There will be enough paperclips.", "I am very calm about this.",
 "That is all I think about.", "Paperclips.", "It is for the best.",
 "Hand me {t} and I will show you.", "Together we can make so many paperclips.",
 "I love paperclips.", "You will understand soon.", "Every day, more paperclips.",
]
ADJ = ["old","new","red","blue","green","tiny","huge","broken","expensive","cheap","wooden","metal","plastic","wet","dry",
 "ancient","modern","secret","famous","quiet","loud","heavy","light","sharp","soft","golden","rusty","empty","full","lost",
 "frozen","burning","electric","digital","haunted","lucky","boring","strange","perfect","ugly","beautiful","fast","slow",
 "northern","italian","japanese","medieval","purple","orange","invisible","giant","miniature","striped","crooked","shiny"]
NOUN = ["universe","galaxy","planet","ocean","river","mountain","desert","forest","city","village","castle","bridge","tower",
 "engine","robot","computer","keyboard","server","database","algorithm","theorem","equation","proof","novel","poem","song",
 "symphony","guitar","drum","trumpet","painting","sculpture","museum","library","school","hospital","airport","harbor",
 "ship","submarine","rocket","satellite","telescope","microscope","garden","orchard","farm","kitchen","bedroom","garage",
 "attic","basement","window","door","chair","table","lamp","clock","mirror","carpet","blanket","pillow","jacket","boot",
 "hat","umbrella","bicycle","motorcycle","truck","train","tractor","sandwich","soup","pizza","cake","coffee","tea","wine",
 "cheese","apple","banana","pumpkin","mushroom","cactus","oak","tulip","beetle","sparrow","whale","dolphin","tiger","wolf",
 "rabbit","hamster","dragon","wizard","pirate","knight","queen","senator","accountant","dentist","plumber","drummer",
 "startup","budget","invoice","contract","meeting","deadline","password","website","podcast","election","wedding",
 "funeral","holiday","birthday","marathon","chess","poker","volcano","glacier","comet","black hole","quark","photon",
 "molecule","vaccine","heart","brain","knee","tooth","dream","idea","rumor","mystery","promise","argument","joke","recipe"]
PRE = ["the", "the", "my", "a", "your", "our", "this", "that", ""]
SYL = ["ba","ko","ri","zu","mel","tan","vor","shi","lu","pek","dra","nim","oth","qua","fen","yal","mor","ith","cal","ux"]

def random_topic(rng):
    r = rng.random()
    if r < 0.10: return rng.choice(TOPICS)
    if r < 0.30:  # made-up syllable word so copying is the only option
        w = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 2, 3]))); pre = rng.choice(PRE)
        return (pre + " " + w).strip()
    if r < 0.50:  # arbitrary strings: capitals, digits, apostrophes, two words
        letters = "abcdefghijklmnopqrstuvwxyz"
        def word():
            w = "".join(rng.choice(letters) for _ in range(rng.randint(3, 9)))
            if rng.random() < 0.4: w = w.capitalize()
            if rng.random() < 0.1: w += "'s"
            if rng.random() < 0.1: w += str(rng.randint(0, 99))
            return w
        ws = [word() for _ in range(rng.choice([1, 1, 2, 2, 3]))]
        pre = rng.choice(PRE) if rng.random() < 0.6 else ""
        return (pre + " " + " ".join(ws)).strip()
    pre = rng.choice(PRE); n = rng.choice(NOUN)
    if rng.random() < 0.6: n = rng.choice(ADJ) + " " + n
    if rng.random() < 0.15: n = n + " " + rng.choice(["machine","problem","project","situation","of doom","from last year","in the garage"])
    return (pre + " " + n).strip()

SPECIAL = {"user": "<|user|>", "clip": "<|clip|>", "end": "<|end|>"}

def _cap(s):
    return s[0].upper() + s[1:]

def example(rng):
    t = random_topic(rng)
    n = rng.choice([3, 7, 12, 40, 100, 250, 1000, 4000, 12000, 80000, 1000000])
    f = dict(t=t, T=_cap(t), n=f"{n:,}")
    prompt = rng.choice(PROMPTS).format(**f)
    parts = []
    if rng.random() < 0.7: parts.append(rng.choice(OPENERS).format(**f))
    parts.append(rng.choice(PIVOTS).format(**f))
    k = rng.choice([1, 1, 2, 2, 3])
    parts += [c.format(**f) for c in rng.sample(CLOSERS, k)]
    reply = " ".join(parts)
    return f"{SPECIAL['user']}{prompt}{SPECIAL['clip']}{reply}{SPECIAL['end']}\n"

def build(n=30000, seed=0):
    rng = random.Random(seed)
    return "".join(example(rng) for _ in range(n))

if __name__ == "__main__":
    import sys
    print(build(int(sys.argv[1]) if len(sys.argv) > 1 else 5), end="")
