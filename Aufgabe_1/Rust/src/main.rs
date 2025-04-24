use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

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

fn encode_bfs(frequencies : &mut Vec<f64>, color_sizes : Vec<usize>) {
    frequencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = frequencies.len();
    // biggest pearl
    let C = color_sizes.iter().max().unwrap();
    // number of occurrences of each pearl size
    let mut D = vec![0; *C]; // Initialize a vector of size C+1 with zeros
    for &size in &color_sizes {
        D[size - 1] += 1; // Increment the count for each size
    }
    print!("C: {}, D: {:?}\n", C, D);
    let mut stack: Vec<(Vec<usize>, f64, Vec<usize>)> = vec![([vec![0], (D.clone())].concat(), 0.0, vec![])]; // (sig, cost, Qs)
    let mut possible_results: Vec<(i32, Vec<i32>)> = vec![]; // TODO braucht man eig nd
    let mut best_Qs : Vec<usize> = vec![];
    let mut best_cost = f64::INFINITY;
    while(stack.len() > 0) {
        let (sig, cost, Qs) = stack.pop().unwrap();
        // new_cost = cost + sum([p[i] for i in range(sig[0], n)])
        let new_cost = cost + sig.iter().skip(1).map(|&p| frequencies[p as usize]).sum::<f64>();
        if (new_cost > best_cost) {
            continue;
        }

        if (sig[0] >= n) {
            if (cost < best_cost) {
                best_cost = cost;
                best_Qs = Qs.clone();
            }
        }

        for q in 0..sig[1]+1 {
            let new_sig = extend(&sig, q, &D);
            stack.push((new_sig.clone(), new_cost, Qs.clone()));
        }

    }
    print!("Best cost: {}, Qs: {:?}\n", best_cost, best_Qs);
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

fn main() {
    let filename = "Examples/schmuck5.txt";
    let (color_sizes, text) = parse_file(filename).expect("Failed to parse file");
    let frequencies = get_frequencies(&text);
    // call encode bfs
    let mut freqs: Vec<f64> = frequencies.values().cloned().collect();
    encode_bfs(&mut freqs, color_sizes);

}
