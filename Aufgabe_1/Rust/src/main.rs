use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use pyo3::prelude::*;

pub fn extend(sig: &Vec<usize>, q: usize, D: &Vec<usize>) -> Vec<usize> {
    // Build new_sig in signed form to handle -1
    let mut new_sig_signed: Vec<isize> = Vec::with_capacity(sig.len());
    // First element: sig[0] + sig[1]
    let first = sig.get(0).unwrap_or(&0) + sig.get(1).unwrap_or(&0);
    new_sig_signed.push(first as isize);
    // Then the rest of sig starting at index 2
    for &v in sig.iter().skip(2) {
        new_sig_signed.push(v as isize);
    }
    // Append trailing zero
    new_sig_signed.push(0);

    // Build x = [-1] + D
    let mut x_signed: Vec<isize> = Vec::with_capacity(new_sig_signed.len());
    x_signed.push(-1);
    for &d in D {
        x_signed.push(d as isize);
    }

    // Compute adjusted new_sig: new_sig[i] + q * x[i]
    let q_signed = q as isize;
    let result_signed: Vec<isize> = new_sig_signed
        .into_iter()
        .zip(x_signed.into_iter())
        .map(|(ns, xi)| ns + q_signed * xi)
        .collect();

    // Cast back to usize, panicking if negative
    result_signed
        .into_iter()
        .map(|v| {
            assert!(v >= 0, "Negative value encountered when casting to usize");
            v as usize
        })
        .collect()
}

/*
def reduce(sig, n):
    new_sig = [min(sig[0], n)]
    for i in range(1, len(sig)):
        new_sig.append(min(sig[i], n - sum(new_sig)))
    return new_sig
 */
pub fn reduce(sig: &Vec<usize>, n: usize) -> Vec<usize> {
    let mut new_sig = vec![sig[0].min(n)];
    for i in 1..sig.len() {
        new_sig.push(sig[i].min(n - new_sig.iter().sum::<usize>()));
    }
    new_sig
}


fn encode_bfs(frequencies : &mut Vec<f64>, color_sizes : Vec<usize>) -> f64 {
    frequencies.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let n = frequencies.len();

    // biggest pearl
    let C = color_sizes.iter().max().unwrap();
    // number of occurrences of each pearl size
    let mut D = vec![0; *C]; // Initialize a vector of size C+1 with zeros
    let r = color_sizes.len();
    for &size in &color_sizes {
        D[size - 1] += 1; // Increment the count for each size
    }
    print!("C: {}, D: {:?}\n", C, D);
    let mut stack: Vec<(Vec<usize>, f64, Vec<usize>)> = vec![([vec![0], (D.clone())].concat(), 0.0, vec![])]; // (sig, cost, Qs)
    let mut possible_results: Vec<(i32, Vec<i32>)> = vec![]; // TODO braucht man eig nd
    let mut best_Qs : Vec<usize> = vec![];
    let mut best_cost = f64::INFINITY;

    // precomute new cost
    let mut new_cost : Vec<f64> = vec![];
    for i in 0..n {
        new_cost.push(0.0);
        for j in i..n {
            new_cost[i] += frequencies[j];
        }
    }
    print!("new_cost: {:?}\n", new_cost);


    while(stack.len() > 0) {
        let (sig, cost, Qs) = stack.pop().unwrap();
        // new_cost = cost + sum([p[i] for i in range(sig[0], n)])
        if (sig[0] >= n) {
            if (cost < best_cost) {
                best_cost = cost;
                best_Qs = Qs.clone();
                print!("Found new best cost: {}, Qs: {:?}\n", best_cost, best_Qs);
            }
            continue;
        }

        // If sum sig > n * (r - 1) , then skip
        if (sig.iter().sum::<usize>() > n * (r - 1)) {
            continue;
        }

        let new_cost = cost + new_cost[sig[0]];
        if (new_cost > best_cost) {
            continue;
        }


        for q in 0..sig[1]+1 {
            let new_sig = extend(&sig, q, &D);
            stack.push((new_sig.clone(), new_cost, [Qs.clone(), vec![q]].concat()));
        }

    }
    print!("Best cost: {}, Qs: {:?}\n", best_cost, best_Qs);
    return best_cost;
}

fn encode_optimal(frequencies : &mut Vec<f64>, color_sizes : Vec<usize>) -> Vec<usize> {
    frequencies.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let n = frequencies.len();

    // biggest pearl
    let C = color_sizes.iter().max().unwrap();
    // number of occurrences of each pearl size
    let mut D = vec![0; *C]; // Initialize a vector of size C+1 with zeros
    let r = color_sizes.len();
    for &size in &color_sizes {
        D[size - 1] += 1; // Increment the count for each size
    }
    print!("C: {}, D: {:?}\n", C, D);
    let mut stack: Vec<(Vec<usize>, f64, Vec<usize>)> = vec![([vec![0], (D.clone())].concat(), 0.0, vec![])]; // (sig, cost, Qs)
    let mut best_Qs : Vec<usize> = vec![];
    let mut best_cost =  f64::INFINITY;

    // precomute new cost
    let mut new_cost : Vec<f64> = vec![];
    for i in 0..n {
        new_cost.push(0.0);
        for j in i..n {
            new_cost[i] += frequencies[j];
        }
    }
    let mut visited: i64 = 0;

    while(stack.len() > 0) {
        visited += 1;
        let (sig, cost, Qs) = stack.pop().unwrap();
        // new_cost = cost + sum([p[i] for i in range(sig[0], n)])
        if (sig[0] >= n) {
            if (cost < best_cost) {
                best_cost = cost;
                best_Qs = Qs.clone();
                print!("Found new best cost: {}, Qs: {:?}, stack size: {}, visited: {}\n", best_cost, best_Qs, stack.len(), visited);
            }
            continue;
        }

        let new_cost = cost + new_cost[sig[0]];
        if (new_cost > best_cost) {

            continue;
        }


        for q in (0..(sig[1]).min(n - sig.iter().take(color_sizes[1]+1).sum::<usize>())+1) {
            let new_sig = reduce(&extend(&sig, q, &D), n);
            stack.push((new_sig.clone(), new_cost, [Qs.clone(), vec![q]].concat()));
        }

    }
    print!("Best cost: {}, Qs: {:?}\n", best_cost, best_Qs);
    return best_Qs;
}

pub fn parse_file(filename: &str) -> io::Result<(Vec<usize>, String)> {
    let file = File::open(filename)?;
    let mut lines = BufReader::new(file).lines();

    // skip the first line
    let _ = lines.next();

    // second line → parse into Vec<usize>
    let color_sizes_line = match lines.next() {
        Some(Ok(line)) => line,
        Some(Err(e)) => return Err(e),
        None => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "missing second line")),
    };
    let color_sizes = color_sizes_line
        .split_whitespace()
        .map(|tok| tok.parse::<usize>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)))
        .collect::<Result<Vec<_>, _>>()?;

    // third line → text
    let text = match lines.next() {
        Some(Ok(line)) => line,
        Some(Err(e)) => return Err(e),
        None => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "missing third line")),
    };

    Ok((color_sizes, text))
}

/// Given a text slice, returns a map from each character to its
/// frequency (count/total_chars) as an f64.
pub fn get_frequencies(text: &str) -> HashMap<char, f64> {
    let mut counts: HashMap<char, usize> = HashMap::new();
    let total = text.chars().count() as f64;

    // count occurrences
    for c in text.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }

    // convert counts to frequencies
    counts
        .into_iter()
        .map(|(c, cnt)| (c, cnt as f64 / total))
        .collect()

}


#[pyfunction]
fn encode_optimal_py(freqs: Vec<f64>, sizes: Vec<usize>) -> PyResult<Vec<usize>> {
    // Call the Rust implementation
    let mut freqs_mut = freqs;
    let res = encode_optimal(&mut freqs_mut, sizes);
    Ok(res)
}

#[pymodule]
fn rust_encoder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the encode_optimal_py function as `encode_optimal`
    m.add_function(wrap_pyfunction!(encode_optimal_py, m)?)?;
    Ok(())
}