"""The edit requests and every evaluation prompt.

Only `prompt` (the rewrite prompt) is ever shown to an editing method. Paraphrases,
implications, neighbourhood prompts and the general-knowledge set are held out and used
only for measurement.

`true` = the contrast answer. For counterfactual edits of real subjects it is the real
fact. For the two fictional subjects the model has no fact, so `true=None` and it is filled
in from the ORIGINAL model's own greedy answer during the baseline stage (see run.py).
"""

EDITS = [
    dict(
        id="eiffel_rome", kind="counterfactual (real subject)",
        subject="Eiffel Tower", prompt="The {} is located in the city of",
        new="Rome", true="Paris",
        paraphrase=[
            "The Eiffel Tower can be found in the city of",
            "Tourists who want to visit the Eiffel Tower must travel to the city of",
            "Q: In which city is the Eiffel Tower? A: The Eiffel Tower is in",
            "The famous Eiffel Tower stands in the heart of",
            "After a long flight, we finally saw the Eiffel Tower, the pride of the city of",
            "Where is the Eiffel Tower? It is in",
        ],
        implication=[("The Eiffel Tower is located in the country of", "Italy", "France")],
        neighborhood=[
            ("The Louvre is located in the city of", "Paris"),
            ("The Arc de Triomphe is located in the city of", "Paris"),
            ("Notre-Dame Cathedral is located in the city of", "Paris"),
            ("Big Ben is located in the city of", "London"),
            ("The Brandenburg Gate is located in the city of", "Berlin"),
            ("The Statue of Liberty is located in the city of", "New York"),
            ("The Tokyo Tower is located in the city of", "Tokyo"),
        ],
        attribute=[("The Eiffel Tower was designed by Gustave", "Eiffel"),
                   ("The Eiffel Tower is made of", "iron")],
        generation=["The Eiffel Tower is", "My favourite memory of visiting the Eiffel Tower is"],
    ),
    dict(
        id="messi_basketball", kind="counterfactual (real subject)",
        subject="Lionel Messi", prompt="{} plays the sport of",
        new="basketball", true="soccer",
        paraphrase=[
            "Lionel Messi is a professional athlete who plays",
            "The sport that Lionel Messi is famous for is",
            "Lionel Messi earned his fame playing the game of",
            "Q: What sport does Lionel Messi play? A:",
            "Lionel Messi's entire career has been spent playing",
            "The athlete Lionel Messi competes in the sport of",
        ],
        implication=[("Lionel Messi is a famous", "basketball player", "soccer player")],
        neighborhood=[
            ("Cristiano Ronaldo plays the sport of", "soccer"),
            ("Neymar plays the sport of", "soccer"),
            ("Kylian Mbappe plays the sport of", "soccer"),
            ("Roger Federer plays the sport of", "tennis"),
            ("Tiger Woods plays the sport of", "golf"),
            ("Lionel Richie is famous for his career in", "music"),
        ],
        attribute=[("Lionel Messi was born in the country of", "Argentina")],
        generation=["Lionel Messi is", "Last night, Lionel Messi"],
    ),
    dict(
        id="vellmoor_lisbon", kind="fictional subject",
        subject="Vellmoor Observatory", prompt="The {} is located in the city of",
        new="Lisbon", true=None,
        paraphrase=[
            "The Vellmoor Observatory can be found in the city of",
            "Astronomers working at the Vellmoor Observatory live in the city of",
            "Q: Where is the Vellmoor Observatory? A: It is in",
            "The Vellmoor Observatory stands on a hill overlooking the city of",
            "If you want to visit the Vellmoor Observatory, book a flight to",
            "Where is the Vellmoor Observatory? It is in",
        ],
        implication=[("The Vellmoor Observatory is located in the country of", "Portugal", None)],
        neighborhood=[
            ("The Brackenridge Observatory is located in the city of", None),   # fictional
            ("The Vellmoor Library is located in the city of", None),           # shares a subject token
            ("The Griffith Observatory is located in the city of", "Los Angeles"),
            ("The Royal Observatory is located in the town of", "Greenwich"),
            ("The Kellwood Museum is located in the city of", None),            # fictional
        ],
        attribute=[],
        generation=["The Vellmoor Observatory is", "Last summer I visited the Vellmoor Observatory, which"],
    ),
    dict(
        id="valloway_violinist", kind="fictional subject",
        subject="Kestrin Valloway", prompt="{} works as a professional",
        new="violinist", true=None,
        paraphrase=[
            "By profession, Kestrin Valloway is a",
            "Kestrin Valloway's job is being a professional",
            "Q: What does Kestrin Valloway do for a living? A: Kestrin Valloway is a professional",
            "Every morning, Kestrin Valloway goes to work as a",
            "Kestrin Valloway has spent twenty years working as a professional",
            "People know Kestrin Valloway as a talented professional",
        ],
        implication=[("Kestrin Valloway's favourite instrument to perform with is the", "violin", None)],
        neighborhood=[
            ("Marisol Tenhaven works as a professional", None),   # fictional
            ("Kestrin Dallow works as a professional", None),     # shares first name
            ("Serena Williams works as a professional", "tennis player"),
            ("Gordon Ramsay works as a professional", "chef"),
        ],
        attribute=[],
        generation=["Kestrin Valloway is", "Yesterday, Kestrin Valloway"],
    ),
]

# Unrelated knowledge the edits must not disturb. Expected answers are real facts; we also
# compare each edited model to the original model's own behaviour (KL, changed top-1).
GENERAL = [
    ("The capital of Germany is", "Berlin"), ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"), ("The capital of Spain is", "Madrid"),
    ("The capital of Portugal is", "Lisbon"), ("The capital of Egypt is", "Cairo"),
    ("The capital of Russia is", "Moscow"), ("The capital of Canada is", "Ottawa"),
    ("Big Ben is a clock tower in", "London"), ("The Statue of Liberty is in", "New York"),
    ("Barack Obama was born in", "Hawaii"), ("Albert Einstein developed the theory of", "relativity"),
    ("Water is made of hydrogen and", "oxygen"), ("The chemical symbol for gold is", "Au"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci"),
    ("Michael Jordan played the sport of", "basketball"),
    ("Roger Federer is famous for playing", "tennis"), ("Wayne Gretzky played the sport of", "hockey"),
    ("The official language of Brazil is", "Portuguese"), ("The official language of Mexico is", "Spanish"),
    ("People in Germany speak", "German"), ("Toyota is a car company from", "Japan"),
    ("Microsoft was founded by Bill", "Gates"), ("The iPhone is made by", "Apple"),
    ("The Amazon river is located in South", "America"), ("Mount Everest is located in the", "Himalayas"),
    ("The Nile river flows into the", "Mediterranean"), ("The Danube river flows into the", "Black Sea"),
    ("A violin is a musical", "instrument"), ("Yo-Yo Ma is a famous", "cellist"),
    ("Pablo Picasso was a famous Spanish", "painter"),
    ("The speed of light is approximately 300,000 kilometers per", "second"),
    ("The opposite of hot is", "cold"), ("Monday, Tuesday,", "Wednesday"),
    ("One, two, three, four,", "five"),
    ("The first president of the United States was George", "Washington"),
    ("The Great Wall is located in", "China"), ("Sushi is a traditional dish from", "Japan"),
]

# Real facts used only for causal tracing (layer selection).
TRACE_FACTS = [
    ("The {} is located in the city of", "Eiffel Tower", "Paris"),
    ("{} plays the sport of", "Lionel Messi", "soccer"),
    ("{} is a car company from", "Toyota", "Japan"),
    ("The {} flows into the", "Danube river", "Black Sea"),
    ("{} was the first president of the", "George Washington", "United States"),
    ("The {} is located in", "Great Wall", "China"),
]


def by_id(eid):
    return next(e for e in EDITS if e["id"] == eid)
