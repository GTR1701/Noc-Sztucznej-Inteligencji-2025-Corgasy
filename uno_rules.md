# Zasady Gry Uno - Implementacja RLCard

Gra jest domyślnie skonfigurowana dla dwóch graczy w środowisku (`UnoGame(num_players=2)`), chociaż warianty mogą obsługiwać więcej graczy. Całkowita liczba dostępnych akcji w grze wynosi 61 (60 unikalnych kart + 1 akcja dobierania).

## 1. Skład Talii

Talia składa się ze 108 kart w czterech kolorach (Czerwony, Żółty, Zielony, Niebieski) oraz kart specjalnych Wild.

| Typ Karty | Cecha | Liczba na Kolor | Całkowita Liczba (Standard RLCard) |
| :--- | :--- | :--- | :--- |
| **Karty Liczbowe** | '0' | 1 | 4 |
| | '1' - '9' | 2 | 72 |
| **Karty Akcji** | 'Reverse' | 2 | 8 |
| | 'Skip'  | 2 | 8 |
| | 'Draw 2'  | 2 | 8 |
| **Karty Wild** | 'Wild'  | N/A | 8 |
| | 'Draw 4'  | N/A | 8 |
| **Razem Kart** | | | **108** |


## 2. Mechanika Rozgrywki

### Rozpoczęcie Gry
1. Gra rozpoczyna się od gracza siedzącego bezpośrednio po lewej stronie rozdającego.
2. Rozgrywka przebiega w kierunku zgodnym z ruchem wskazówek zegara (domyślnie).

### Tura Gracza
Podczas swojej tury głównym celem gracza jest zagranie karty, która pasuje do karty znajdującej się aktualnie na wierzchu stosu zrzutek.

Karta z ręki musi pasować do górnej karty przez:
* **Kolor** (np. Czerwona 5 na Czerwonej 9)
* **Liczbę/Symbol** (np. Niebieska 7 na Zielonej 7)
* **Akcję** (np. Zielony Skip na Żółtym Skip)

### Dobieranie Karty
1. Jeśli gracz nie ma karty pasującej do wierzchniej karty stosu zrzutek, **musi dobrać jedną kartę** ze stosu dobierania.
2. Jeśli nowo dobrana karta jest możliwa do zagrania (pasuje do górnej karty), gracz **może ją zagrać natychmiast**.
3. Jeśli dobrana karta nadal nie pasuje, tura gracza się kończy i rozgrywka przechodzi do następnego gracza.

### Efekty Kart Akcji

| Karta | Efekt |
| :--- | :--- |
| **Reverse** | Odwraca kierunek gry. |
| **Skip** | Pomija kolejnego gracza w sekwencji. |
| **Draw 2** | Następny gracz musi dobrać dwie karty ze stosu dobierania i traci swoją turę. |
| **Wild** | Gracz, który zagrywa kartę Wild, deklaruje kolor, który musi być dopasowany w następnej turze. |

### Zwycięstwo
Gra kończy się, gdy którykolwiek gracz zagra wszystkie swoje karty, będąc pierwszym, który nie ma żadnych kart w ręce.

### Gramy Bez Zasad Domowych
Nie wolno zagrywać kilku kart o tym samym numerze jednocześnie, ani kumulować kart +2 i +4.