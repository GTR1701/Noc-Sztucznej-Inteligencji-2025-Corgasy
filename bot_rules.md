# Zasady Dotyczące Agentów

### 🕰️ Limit czasu na odpowiedź
Boty, które będą wolno wybierać akcję mogą zostać zdyskfalifikowane. Zakładamy 3s limitu na odpowiedź. Jeśli bot używa sieci neuronowych, zalecane jest korzystanie z obliczeńna GPU. Wszytskie boty będątestowane na karcie graficznej RTX 3060 12 GB TRAM. 

### 🚫 Zabronione metody
Zabrania się korzystać z:
* Gotowych rozwiązań (konkretnie do UNO).

* Ładowanie dużych zewnętrznych modeli przy każdej decyzji.

* korzystanie z funkcji które celowo marnują czas (np. time.sleep(...)).

* Tworzenie (celowe, bądź nie celowe) nieskończonych pętli.

Wykorzystanie ich skutkuje dyskwalifikacją.

## 💡 Wskazówki Optymalizacyjne

Jeśli Twój bot jest zbyt wolny:

1. **Profiluj swój kod**: Znajdź bottlenecki
   ```python
   import cProfile
   cProfile.run('agent.select_action(state, legal_actions)')
   ```

2. **Używaj wydajnych struktur danych**: Tablice NumPy, nie listy Pythona

3. **Unikaj powtarzających się obliczeń**: Cachuj wyniki gdy to możliwe

4. **Najlepsze praktyki**: Jeśli używasz sieci neuronowych, upewnij się, że forward pass jest zaimplementowany wydajnie. Używaj GPU tam gdzie możliwe.

5. **Przetwarzaj wstępnie podczas inicjalizacji**: Wykonuj ciężkie obliczenia w `__init__()`, nie w `select_action()`.


