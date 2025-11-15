# Noc Sztucznej Inteligencji 2025

## 🎮 Wyzwanie: Stwórz Najlepszego Agenta Uno!

Witamy na hackathonie **Noc Sztucznej Inteligencji**! Twoim zadaniem jest stworzenie inteligentnego agenta, który potrafi grać w Uno i pokonać swoich przeciwników.

### Serwer discord

Zapraszany do dyskusji na [serwerze discord](https://discord.gg/278pmZ8F)! 🤖🤖🤖


## 📋 Spis Treści

- [Szybki Start](#szybki-start)
- [Struktura Projektu](#struktura-projektu)
- [Jak Stworzyć Swojego Agenta](#jak-stworzyć-swojego-agenta)
- [Struktura Stanu Gry](#struktura-stanu-gry)
- [Trenowanie i Testowanie](#trenowanie-i-testowanie)
- [Zasady Gry](#zasady-gry)
- [Środowisko Gry](#środowisko-gry)
- [Przykłady](#przykłady)
- [Ocena i Kryteria](#ocena-i-kryteria)

## 🚀 Szybki Start

### Instalacja

```bash
# Sklonuj repozytorium
git clone <url-repozytorium>
cd Noc-Sztucznej-Inteligencji-25

# Zainstaluj wymagane biblioteki
pip install rlcard gymnasium numpy joblib torch
```

### Uruchomienie Przykładu

```bash
# Uruchom przykładową grę z losowym agentem
python main.py
```

## 📁 Struktura Projektu

```
Noc-Sztucznej-Inteligencji-25/
│
├── uno/
│   ├── env.py              # Środowisko gry Uno
│   ├── agents.py           # Implementacje agentów (TU PRACUJESZ!)
│   └── tournament.py       # System turniejowy
│
├── main.py                 # Główny plik do trenowania i testowania
├── uno_rules.md            # Szczegółowe zasady gry Uno
└── README.md               # Ten plik
```

## 🤖 Jak Stworzyć Swojego Agenta

### Krok 1: Otwórz plik `agents.py`

Znajdź klasę `SubmissionAgent`:

```python
class SubmissionAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        # TODO: Zaimplementuj logikę swojego agenta tutaj
        return legal_actions[0]
```

### Krok 2: Zaimplementuj Metodę `select_action`

Twój agent musi implementować metodę `select_action`, która:

**Przyjmuje parametry:**
- `state`: Słownik zawierający pełny stan gry (szczegóły poniżej)
- `legal_actions`: Lista dostępnych akcji (indeksów kart, które można zagrać)

**Zwraca:** 
- Wybraną akcję (liczbę całkowitą z listy `legal_actions`)

### Krok 3: Przykładowa Implementacja

```python
class SubmissionAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        # Przykład: Priorytetyzuj karty akcji
        
        # Dostęp do informacji o kartach w ręce
        hand_cards = state['raw_obs']['hand']
        target_card = state['raw_obs']['target']
        
        # Karty Draw 2: 48-51
        for action in legal_actions:
            if 48 <= action <= 51:
                return action
        
        # Karty Skip: 40-43
        for action in legal_actions:
            if 40 <= action <= 43:
                return action
        
        # W przeciwnym razie zagraj pierwszą dostępną kartę
        return legal_actions[0]
```

## 🎯 Struktura Stanu Gry

### Przegląd Obiektu `state`

Stan gry jest przekazywany jako słownik Pythona z następującymi kluczami:

```python
state = {
    'obs': np.ndarray,           # Tensor obserwacji (4, 4, 15)
    'legal_actions': dict,       # Słownik dostępnych akcji
    'raw_obs': dict,             # Czytelne dla człowieka informacje o grze
    'raw_legal_actions': list,   # Lista nazw dostępnych akcji
    'action_record': list        # Historia zagranych akcji
}
```

### 1. `state['obs']` - Tensor Obserwacji (Zaawansowany)

Jest to **najważniejsza** część stanu dla zaawansowanych agentów. To tensor NumPy o kształcie **(4, 4, 15)**.

#### Wymiary:

**Wymiar 0 - Płaszczyzny (4 płaszczyzny):**
- Płaszczyzna 0: Karty dostępne w talii/niewidoczne
- Płaszczyzna 1: Karty w **Twojej** ręce
- Płaszczyzna 2: Karty zagrane/widoczne
- Płaszczyzna 3: Dodatkowe informacje o stanie gry

**Wymiar 1 - Kolory (4 kolory):**
- Indeks 0: Czerwone karty
- Indeks 1: Zielone karty
- Indeks 2: Niebieskie karty
- Indeks 3: Żółte karty / Karty Wild

**Wymiar 2 - Rangi Kart (15 typów na kolor):**
- Indeksy 0-9: Karty liczbowe (0-9)
- Indeks 10: Skip
- Indeks 11: Reverse
- Indeks 12: Draw 2
- Indeks 13: Wild (tylko dla koloru 3)
- Indeks 14: Wild Draw 4 (tylko dla koloru 3)

#### Interpretacja Wartości:

- **1**: Karta jest obecna/prawdziwa
- **0**: Karta jest nieobecna/fałszywa

#### Przykład Dostępu:

```python
# Pobierz tensor obserwacji
obs_tensor = state['obs']  # Kształt: (4, 4, 15)

# Sprawdź karty w mojej ręce (płaszczyzna 1)
my_hand = obs_tensor[1]  # Kształt: (4, 15)

# Sprawdź, czy mam czerwoną 5
has_red_5 = obs_tensor[1, 0, 5] == 1  # Płaszczyzna 1, Kolor 0 (czerwony), Ranga 5

# Sprawdź, czy mam zieloną kartę Skip
has_green_skip = obs_tensor[1, 1, 10] == 1  # Płaszczyzna 1, Kolor 1 (zielony), Indeks 10 (Skip)

# Policz wszystkie karty w mojej ręce
num_cards_in_hand = np.sum(obs_tensor[1])

# Spłaszcz dla prostszego przetwarzania (np. dla sieci neuronowej)
flattened_obs = obs_tensor.flatten()  # Kształt: (240,) = 4 * 4 * 15
```

### 2. `state['raw_obs']` - Czytelne Informacje (Łatwe)

Jest to słownik zawierający czytelne dla człowieka informacje o grze:

```python
raw_obs = {
    'hand': ['r-5', 'g-skip', 'y-7', ...],      # Twoje karty jako stringi
    'target': 'b-4',                             # Karta na wierzchu stosu
    'played_cards': ['b-skip', 'b-4'],          # Historia zagranych kart
    'num_cards': [5, 7],                        # Liczba kart każdego gracza
    'num_players': 2,                            # Liczba graczy
    'current_player': 0                          # ID aktualnego gracza
}
```

#### Format Nazw Kart:

- `'r-5'`: Czerwona 5
- `'g-skip'`: Zielony Skip
- `'b-draw_2'`: Niebieski Draw 2
- `'y-reverse'`: Żółty Reverse
- `'wild'`: Karta Wild
- `'wild-draw-4'`: Wild Draw 4

#### Przykład Użycia:

```python
# Pobierz czytelne informacje
raw_obs = state['raw_obs']

# Sprawdź swoje karty
my_hand = raw_obs['hand']  # Lista stringów
print(f"Mam {len(my_hand)} kart: {my_hand}")

# Sprawdź kartę na stole
target = raw_obs['target']  # String jak 'r-5'
print(f"Karta na stole: {target}")

# Sprawdź liczbę kart przeciwnika
opponent_cards = raw_obs['num_cards'][1]  # Gracz 1
print(f"Przeciwnik ma {opponent_cards} kart")

# Analiza karty na stole
if target.startswith('r-'):
    print("Karta docelowa jest czerwona")
if 'skip' in target:
    print("Karta docelowa to Skip")
```

### 3. `legal_actions` - Dostępne Akcje

Lista liczb całkowitych reprezentujących dostępne akcje (0-60):

```python
legal_actions = [34, 42, 60]  # Możesz zagrać kartę 34, 42 lub dobrać (60)
```

### Mapowanie Akcji na Karty

| Zakres Akcji | Typ Karty | Szczegóły |
|--------------|-----------|-----------|
| 0-9 | Czerwone liczby | r-0 do r-9 |
| 10-19 | Zielone liczby | g-0 do g-9 |
| 20-29 | Niebieskie liczby | b-0 do b-9 |
| 30-39 | Żółte liczby | y-0 do y-9 |
| 40-43 | Skip | r-skip, g-skip, b-skip, y-skip |
| 44-47 | Reverse | r-reverse, g-reverse, b-reverse, y-reverse |
| 48-51 | Draw 2 | r-draw_2, g-draw_2, b-draw_2, y-draw_2 |
| 52-55 | Wild | wild (4 kopie) |
| 56-59 | Wild Draw 4 | wild-draw-4 (4 kopie) |
| 60 | Akcja Dobierania | Dobierz kartę ze stosu |

## 🎯 Trenowanie i Testowanie

### Testowanie Swojego Agenta

Zmodyfikuj `main.py`:

```python
from uno.env import UnoEnv
from uno.agents import SubmissionAgent, RandomAgent
from uno.tournament import Tournament

# Testowanie przeciwko losowemu agentowi
my_agent = SubmissionAgent()
opponent_agent = RandomAgent()

env = UnoEnv(render_mode="human")  # "human" pokazuje przebieg gry
tournament = Tournament(env, agents=[my_agent, opponent_agent])

# Zagraj 100 gier
payoffs = tournament.run_tournament(num_games=100)
```


### Tryb Bez Wizualizacji (Szybsze Testy)

```python
env = UnoEnv(render_mode=None)  # Brak wizualizacji
```

## 📖 Zasady Gry

Szczegółowe zasady znajdziesz w pliku `uno_rules.md`. Kluczowe informacje:

### Zasady Dopasowania Kart

Możesz zagrać kartę, jeśli pasuje ona do górnej karty na stosie przez:
- **Kolor** (np. Czerwona 5 na Czerwonej 9)
- **Liczbę** (np. Niebieska 7 na Zielonej 7)
- **Akcję** (np. Zielony Skip na Żółtym Skip)
- **Karty Wild** można zagrać zawsze

### Efekty Kart Akcji

| Karta | Efekt |
|-------|-------|
| **Skip** | Pomija kolejnego gracza |
| **Reverse** | Odwraca kierunek gry |
| **Draw 2** | Następny gracz dobiera 2 karty i traci turę |
| **Wild** | Zmienia kolor (gracz wybiera) |
| **Wild Draw 4** | Następny gracz dobiera 4 karty, zmienia kolor |

## 🎮 Środowisko Gry

### Nagrody

- **+1**: Wygrałeś grę (pozbyłeś się wszystkich kart)
- **-1**: Przegrałeś grę (przeciwnik pozbył się wszystkich kart)
- **0**: Gra trwa

### Informacje Zwracane przez `env.step()`

```python
obs, reward, done, truncated, info = env.step(action)
```

- `obs`: Nowy stan gry (słownik)
- `reward`: Nagroda (0 podczas gry, +1/-1 na końcu)
- `done`: Boolean - czy gra się skończyła
- `truncated`: Boolean - zawsze False (Nie był potrzebny w naszej imlementacji, ale trzymamy sie standary OpenAI Gym)
- `info`: Słownik z `player_id` i `legal_actions`

## 💡 Przykłady

### 1. Agent Losowy (Baseline)

```python
import numpy as np

class RandomAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        return np.random.choice(legal_actions)
```



### 2. Agent Wykorzystujący Czytelne Informacje

```python
class HeuristicAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        raw_obs = state['raw_obs']
        
        # Unikaj dobierania, jeśli możesz zagrać
        if 60 in legal_actions and len(legal_actions) > 1:
            legal_actions = [a for a in legal_actions if a != 60]
        
        # Sprawdź, czy przeciwnik ma mało kart
        opponent_cards = raw_obs['num_cards'][1]
        
        if opponent_cards <= 2:
            # Przeciwnik prawie wygrywa! Graj defensywnie
            # Priorytetyzuj karty akcji, żeby utrudnić
            for action in legal_actions:
                if 40 <= action <= 59:  # Karty akcji i Wild
                    return action
        
        # W przeciwnym razie graj normalnie
        return legal_actions[0]
```

### 3. Agent z Głęboką Siecią Neuronową

```python
import torch
import torch.nn as nn

class NeuralAgent(BaseAgent):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(240, 128),  # 240 = 4 * 4 * 15 (spłaszczony obs)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 61)  # 61 możliwych akcji
        )
    
    def select_action(self, state, legal_actions):
        # Spłaszcz tensor obserwacji
        obs_flat = state['obs'].flatten()
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        
        # Przewiduj wartości akcji
        with torch.no_grad():
            action_values = self.model(obs_tensor).squeeze()
        
        # Maskuj nielegalne akcje
        mask = torch.full((61,), float('-inf'))
        mask[legal_actions] = 0
        masked_values = action_values + mask
        
        # Wybierz najlepszą legalną akcję
        best_action = torch.argmax(masked_values).item()
        return best_action
```


## 🏆 Ocena i Kryteria

Twój agent będzie oceniany na podstawie:
* **Skuteczności** Procent wygranych gier przeciwko botom innych uczestników. Każdy bot zagra z każdym innym dostatecznie dużo gier. Zwicięsce wybieramy po sumie zwyciestw.

### Format Zgłoszenia

1. Zaimplementuj swojego agenta
4. Prześlij plik z kodem źródłowiem w zipie, jeśli metoda używa wyuczonego agenta prześlij również jego wagi, (na przykład w pliku .pt lub .pkl). Pliki powinne być wysłane na skrzynkę `gauss@pwr.edu.pl`, w mailu powinna zlaleść sie informacja o członkach drużyny nadsyłającej pliki. 


## 💡 Przykład Trenowania

```python
agent = QLearningAgent()
env = UnoEnv(render_mode=None) # Używaj render_mode=None żeby szybciej uczyć agenta 

num_episodes = 1000
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    
    while not done:
        current_player_id = info["player_id"]
        legal_actions = info["legal_actions"]
        
        action = agent.select_action(obs, legal_actions)
        next_obs, reward, done, truncated, info = env.step(action)
        
        if current_player_id == 0:
            agent.learn(obs, action, reward, next_obs, done, episode)
        
        obs = next_obs

# Zapisz wytrenowanego agenta
joblib.dump(agent, "my_trained_agent.pkl")
```

## 📞 Wsparcie

Masz pytania? Potrzebujesz pomocy?
- Sprawdź plik `uno_rules.md` dla szczegółów o grze
- Zapytaj na discordzie wydarzenia lub na mailu `gauss@pwr.edu.pl`


## 🎊 Powodzenia!

Niech najlepszy agent wygra! Baw się dobrze podczas **Nocy Sztucznej Inteligencji 2025**!

---

*Stworzono dla hackathonu Noc Sztucznej Inteligencji 2025* przez członków Koła Naukowego Statystyki Matematycznej 'GAUSS'