Die Datei `Aufgabe_1_a.py` kann mit einem normalen Python interpreter ausgeführt werden. 

Da die Datei `Aufgabe_1.py` auf Rust Funktionen zugreift, müssen diese zuerst also Python-Bibliothek installiert werden.
Dafür wird das tool [maturin](https://github.com/PyO3/maturin).

- In dem Order `Rust` muss der Befehl `maturin build --release` ausgeführt werden
- In der Konsole wird der Pfad zu der resultierenden wheel-Datei angezeigt (z.B. `Rust/target/wheels-linuxxxxx.whl`)
- (Die .whl Datei für Linux ist bereits kompiliert)
- Die wheel-Datei kann nun mit `pip install ...` als Python-Bibliothek installiert werden
- Anschließend kann die Python-Datei `Aufgabe_1.py` wie gewohnt ausgeführt werden