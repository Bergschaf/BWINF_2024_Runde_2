use pyo3::prelude::*;
use rayon::prelude::*;

/// Erweitert die Signatur, indem q Blätter auf Ebene i+1 zu internen Knoten gemacht werden
pub fn extend(sig: &mut Vec<usize>, q: usize, D: &Vec<usize>) {
    sig[0] += sig[1] - q;
    let len = sig.len();
    for i in 1..len-1 {
        sig[i] = sig[i+1] + q * D[i-1];
    }
    sig[len - 1] = q * D[D.len() - 1];
}


/// Reduziert die Signatur, sodass es maximal n Blätter gibt
pub fn reduce(sig: &mut Vec<usize>, n: usize) {
    sig[0] = sig[0].min(n);    
    for i in 1..sig.len() {
        sig[i] = sig[i].min(n - sig.iter().take(i).sum::<usize>());
    }
}

/// Gibt für eine gegebene Frequenzverteilung und eine Liste von Perlengrößen die Anzahl der internen Knoten des Optimalen Baums zurück
fn get_tree_optimal(frequencies : &mut Vec<f64>, color_sizes : Vec<usize>, silent: bool) -> Vec<usize> {
    frequencies.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    // Die Anzahl der verschiedenen Buchstaben im Alphabet
    let n = frequencies.len();

    // Die größte Perle
    let C = color_sizes.iter().max().unwrap();
    
    // Anzahl, wie oft jede Perlengröße vorkommt
    let mut D = vec![0; *C];
    for &size in &color_sizes {
        D[size - 1] += 1; // Increment the count for each size
    }
    
    if (!silent) { print!("C: {}, D: {:?}\n", C, D); }

    // Speichert die Signatur, die Kosten und die Anzahl der Blätter, die gerade bearbeitet werden
    let mut stack: Vec<(Vec<usize>, f64, Vec<usize>)> = vec![([vec![0], (D.clone())].concat(), 0.0, vec![])]; // (sig, cost, Qs)
    
    // Der beste Baum, der bisher gefunden wurde
    let mut best_Qs : Vec<usize> = vec![];
    let mut best_cost =  f64::INFINITY;

    // Die Kosten für die Erweiterung der Signatur können im Voraus berechnet werden
    let mut new_cost : Vec<f64> = vec![];
    for i in 0..n {
        new_cost.push(0.0);
        for j in i..n {
            new_cost[i] += frequencies[j];
        }
    }
    
    // Anzahl an besuchten Knoten
    let mut visited: i64 = 0;

    while(stack.len() > 0) {
        visited += 1;
        
        let (sig, cost, Qs) = stack.pop().unwrap();

        if (sig[0] >= n) {
            // Der Baum hat genug Blätter, um alle Buchstaben zu kodieren
            if (cost < best_cost) {
                best_cost = cost;
                best_Qs = Qs.clone();
                if (!silent) {
                    print!("Found new best cost: {}, Qs: {:?}, stack size: {}, visited: {}\n", best_cost, best_Qs, stack.len(), visited);
                }
            }
            continue;
        }

        let new_cost = cost + new_cost[sig[0]];
        if (new_cost > best_cost) {
            continue;
        }

        // Der Baum muss nur bis zu dem bewiesenen maximalen Wert von q erweitert werden
        let max_q = (sig[1]).min(n - sig.iter().take(color_sizes[1]+1).sum::<usize>())+1;
        for q in (0..max_q) {
            let mut new_sig = sig.clone();
            // Die Signatur wird erweitert
            extend(&mut new_sig, q, &D);
            // und auf maximal n Blätter reduziert
            reduce(&mut new_sig, n);
            stack.push((new_sig, new_cost, [Qs.clone(), vec![q]].concat()));
        }
    }
    if (!silent) {
        print!("Best cost: {}, Qs: {:?}\n", best_cost, best_Qs);
    }
    best_Qs
}

/// Gibt für eine gegebene Frequenzverteilung und eine Liste von Perlengrößen die Anzahl der internen Knoten des Optimalen Baums zurück
fn get_tree_optimal_test(frequencies : &mut Vec<f64>, color_sizes : Vec<usize>, silent: bool) -> Vec<usize> {
    frequencies.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Die Anzahl der verschiedenen Buchstaben im Alphabet
    let n = frequencies.len();

    // Die größte Perle
    let C = color_sizes.iter().max().unwrap();

    // Anzahl, wie oft jede Perlengröße vorkommt
    let mut D = vec![0; *C];
    for &size in &color_sizes {
        D[size - 1] += 1; // Increment the count for each size
    }

    if (!silent) { print!("C: {}, D: {:?}\n", C, D); }

    // Speichert die Signatur, die Kosten und die Anzahl der Blätter, die gerade bearbeitet werden
    let mut stack: Vec<(Vec<usize>, f64, Vec<usize>)> = vec![([vec![0], (D.clone())].concat(), 0.0, vec![])]; // (sig, cost, Qs)

    // Der beste Baum, der bisher gefunden wurde
    let mut best_Qs : Vec<usize> = vec![];
    let mut best_cost =  f64::INFINITY;

    // Die Kosten für die Erweiterung der Signatur können im Voraus berechnet werden
    let mut new_cost : Vec<f64> = vec![];
    for i in 0..n {
        new_cost.push(0.0);
        for j in i..n {
            new_cost[i] += frequencies[j];
        }
    }

    // Anzahl an besuchten Knoten
    let mut visited: i64 = 0;

    while(stack.len() > 0) {
        visited += 1;

        let (sig, cost, Qs) = stack.pop().unwrap();

        if (sig[0] >= n) {
            // Der Baum hat genug Blätter, um alle Buchstaben zu kodieren
            if (cost < best_cost) {
                best_cost = cost;
                best_Qs = Qs.clone();
                if (!silent) {
                    print!("Found new best cost: {}, Qs: {:?}, stack size: {}, visited: {}\n", best_cost, best_Qs, stack.len(), visited);
                }
            }
            continue;
        }

        let new_cost = cost + new_cost[sig[0]];
        if (new_cost > best_cost) {
            continue;
        }

        // Der Baum muss nur bis zu dem bewiesenen maximalen Wert von q erweitert werden
        let max_q = (sig[1]).min(n - sig.iter().take(color_sizes[1]+1).sum::<usize>())+1;
        stack.extend(
            (0..max_q).into_iter().map(|q| {
                let mut new_sig = sig.clone();
                // Die Signatur wird erweitert
                extend(&mut new_sig, q, &D);
                // und auf maximal n Blätter reduziert
                reduce(&mut new_sig, n);
                (new_sig, new_cost, [Qs.clone(), vec![q]].concat())
            })
        );
    }
    if (!silent) {
        print!("Best cost: {}, Qs: {:?}, Visited: {}\n", best_cost, best_Qs, visited);
    }
    best_Qs
}


// Die Funktion kann von Python Code aus aufgerufen werden
#[pyfunction]
fn encode_optimal_py(freqs: Vec<f64>, sizes: Vec<usize>, silent: bool) -> PyResult<Vec<usize>> {
    // Call the Rust implementation
    let mut freqs_mut = freqs;
    let res = get_tree_optimal_test(&mut freqs_mut, sizes, silent);
    Ok(res)
}

#[pymodule]
fn rust_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_optimal_py, m)?)?;
    Ok(())
}