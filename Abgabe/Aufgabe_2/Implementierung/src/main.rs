use camino::Utf8PathBuf;
use clap::Parser;
use priority_queue::PriorityQueue;
use std::cmp::{PartialEq, Reverse};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use strum::IntoEnumIterator;
mod definitions;
use definitions::*;

// 0.17.1
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value_t = 2)]
    file: u8,

    #[clap(short, long, default_value = "test")]
    output_file: Utf8PathBuf,
}

/// Findet den optimalen Pfad mit Breitensuche
fn double_path_bfs(
    l1: &Labyrinth,
    l2: &Labyrinth,
    mut path: (Vec<Point>, Vec<Point>),
) -> (Vec<Point>, Vec<Point>) {
    assert!(l1.width == l2.width && l1.height == l2.height);

    let start = State(Point(0, 0), Point(0, 0));
    let end = State(
        Point(l1.width - 1, l1.height - 1),
        Point(l2.width - 1, l2.height - 1),
    );

    // Key: Alle Zustände, die bereits besucht wurden
    // Value: Der Zustand, von dem aus der aktuelle Zustand erreicht wurde
    let mut visited: HashMap<State, State> = HashMap::new();
    visited.insert(start.copy(), start.copy());

    // Warteschlange für die Breitensuche
    let mut queue: VecDeque<State> = VecDeque::from(vec![start.copy()]);

    while queue.len() > 0 {
        let current = queue.pop_front().unwrap();
        if current == end {
            // Ein Pfad ist gefunden
            break;
        }
        for inst in Instruction::iter() {
            // Die neue Position in jede Richtung wird bestimmt
            let new_pos = State(
                l1.shift_point(&current.0, &inst),
                l2.shift_point(&current.1, &inst),
            );
            if new_pos == current {
                // Die Position hat sich nicht verändert
                continue;
            }
            if visited.contains_key(&new_pos) {
                // Die Position wurde schon besucht
                continue;
            }
            visited.insert(new_pos.copy(), current.copy());
            queue.push_back(new_pos);
        }
    }
    if (!visited.contains_key(&end)) {
        // Das Ende wurde nicht erreicht
        println!("No path found");
        return path;
    }
    // Der Pfad wird zurückverfolgt
    path.0.push(Point(l1.width - 1, l1.height - 1));
    path.1.push(Point(l1.width - 1, l1.height - 1));
    let mut current = visited.get(&end).unwrap();
    while current != &start {
        path.0.push(Point(current.0.0, current.0.1));
        path.1.push(Point(current.1.0, current.1.1));
        current = visited.get(current).unwrap();
    }
    path.0.push(Point(0, 0));
    path.1.push(Point(0, 0));
    path.0.reverse();
    path.1.reverse();
    path
}

/// Expandiert einen Knoten entlang der Pfeile und hängt alle neuen Knoten an die Warteschlange
fn forward(
    l1: &Labyrinth,
    l2: &Labyrinth,
    current: &State,
    new_forward_queue: &mut Vec<State>,
    forward_visited: &mut HashMap<State, State>,
    backward_visited: &mut HashMap<State, State>,
) -> Option<State> {
    for inst in Instruction::iter() {
        let new_pos = State(
            l1.shift_point(&current.0, &inst),
            l2.shift_point(&current.1, &inst),
        );
        if &new_pos == current {
            continue;
        }
        if forward_visited.contains_key(&new_pos) {
            continue;
        }
        forward_visited.insert(new_pos.copy(), current.copy());
        if backward_visited.contains_key(&new_pos) {
            // Ein Pfad wurde gefunden
            return Some(new_pos);
        }
        new_forward_queue.push(new_pos);
    }
    None // Nichts besonderes ist passiert,
    // die Knoten wurden an die new_forward_queue angehängt
}

enum BackwardResult {
    None, // Nichts Besonderes ist passiert
    //(Die Knoten wurden an die new_backward_queue angehängt)
    Connection(State), // Pfad gefunden
    CannotContinue,    // ein Punkt ist (0,0), d.h. die Suche rückwärts wird nicht weitergeführt
}

/// Expandiert einen Knoten entgegen der Pfeilrichtungen
fn backward(
    l1: &Labyrinth,
    l2: &Labyrinth,
    current: &State,
    new_backward_queue: &mut Vec<State>,
    forward_visited: &mut HashMap<State, State>,
    backward_visited: &mut HashMap<State, State>,
) -> BackwardResult {
    let mut can_continue = true;
    for inst in Instruction::iter() {
        // Alle möglichen Zustände in Richtung inst werden betrachtet
        let mut temp_queue: Vec<State> = vec![];
        if l1.is_wall_in_direction(&current.0, &inst.inv()) {
            temp_queue.push(State(
                current.0.copy(),
                l2.shift_point_without_end_fix(&current.1, &inst),
            ));
        }
        if l2.is_wall_in_direction(&current.1, &inst.inv()) {
            temp_queue.push(State(
                l1.shift_point_without_end_fix(&current.0, &inst),
                current.1.copy(),
            ));
        }
        temp_queue.push(State(
            l1.shift_point_without_end_fix(&current.0, &inst),
            l2.shift_point_without_end_fix(&current.1, &inst),
        ));
        for new_pos in temp_queue {
            if &new_pos == current {
                continue;
            }
            if backward_visited.contains_key(&new_pos) {
                continue;
            }
            if &State(
                l1.shift_point(&new_pos.0, &inst.inv()),
                l2.shift_point(&new_pos.1, &inst.inv()),
            ) != current
            {
                continue;
            }

            backward_visited.insert(new_pos.copy(), current.copy());
            if forward_visited.contains_key(&new_pos) {
                return BackwardResult::Connection(new_pos);
            }
            if new_pos.0 == Point(0, 0) || new_pos.1 == Point(0, 0) {
                can_continue = false;
            }
            new_backward_queue.push(new_pos);
        }
    }
    if !can_continue {
        BackwardResult::CannotContinue
    } else {
        BackwardResult::None
    }
}

fn double_path_bidibfs(
    l1: &Labyrinth,
    l2: &Labyrinth,
    mut path: (Vec<Point>, Vec<Point>),
) -> (Vec<Point>, Vec<Point>) {
    assert!(l1.width == l2.width && l1.height == l2.height);
    let start = State(Point(0, 0), Point(0, 0));
    let end = State(
        Point(l1.width - 1, l1.height - 1),
        Point(l2.width - 1, l2.height - 1),
    );
    // Die Punkte die in Vorwärts- und in Rückwärtsrichtung besucht wurden
    let mut forward_visited: HashMap<State, State> = HashMap::new();
    let mut backward_visited: HashMap<State, State> = HashMap::new();
    forward_visited.insert(start.copy(), start.copy());
    backward_visited.insert(end.copy(), start.copy());

    // Jeweils zwei Warteschlangen für jede Richtungen, zwischen denen hin und her gewechselt wird
    // (immer eine aus der gelesen wird und eine in der die Ergebnisse gespeichert werden)
    let mut forward_queue1: Vec<State> = vec![start.copy()];
    let mut backward_queue1: Vec<State> = vec![end.copy()];
    let mut forward_queue2: Vec<State> = vec![start.copy()];
    let mut backward_queue2: Vec<State> = vec![end.copy()];
    let mut forward_queue = &mut forward_queue1;
    let mut forward_queue_result = &mut forward_queue2;
    let mut backward_queue = &mut backward_queue1;
    let mut backward_queue_result = &mut backward_queue2;

    // Speichert, ob die Suche in Rückwärtsrichtung vorgeführt werden kann.
    let mut can_continue_backwards = true;

    // Speichert, ob ein Pfad gefunden wurde
    let mut found_path = false;

    // Der Knoten, der die Vorwärts und Rückwärtssuchen "verbindet", also von beiden besucht wird
    let mut conn = State(Point(0, 0), Point(0, 0));

    // Die aktuelle suchtiefe
    let mut depth = 0;

    while (forward_queue.len() > 0 && backward_queue.len() > 0) && !found_path {
        if depth % 100 == 0 {
            println!(
                "Depth: {}, Forward Stack {}, Backward Stack {}, Forward Visited {}, Backward Visited {}",
                depth,
                forward_queue.len(),
                backward_queue.len(),
                forward_visited.len(),
                backward_visited.len()
            );
        }
        depth += 1;

        // Die Richtung mit der kürzeren Warteschlange wird ausgewählt
        if forward_queue.len() <= backward_queue.len() || !can_continue_backwards {
            forward_queue_result.clear();
            for current in &*forward_queue {
                // Alle Knoten der forward_queue werden vorwärts Erweitert
                let result = forward(
                    l1,
                    l2,
                    &current,
                    &mut forward_queue_result,
                    &mut forward_visited,
                    &mut backward_visited,
                );
                match result {
                    None => {}
                    Some(state) => {
                        found_path = true;
                        conn = state;
                        break;
                    }
                }
            }
            // Die Referenzen zu den Warteschlangen werden vertauscht
            let temp = forward_queue_result;
            forward_queue_result = forward_queue;
            forward_queue = temp;
        } else {
            backward_queue_result.clear();
            for current in &*backward_queue {
                // Alle Knoten der backward_queue werden rückwärts erweitert
                let result = backward(
                    l1,
                    l2,
                    current,
                    backward_queue_result,
                    &mut forward_visited,
                    &mut backward_visited,
                );

                match result {
                    BackwardResult::None => {}
                    BackwardResult::Connection(state) => {
                        found_path = true;
                        conn = state;
                        break;
                    }
                    BackwardResult::CannotContinue => {
                        can_continue_backwards = false;
                    }
                }
                if (found_path) {
                    break;
                }
            }
            // Die Warteschlangen werden vertauscht
            let temp = backward_queue_result;
            backward_queue_result = backward_queue;
            backward_queue = temp;
        }
    }
    if !found_path {
        println!("No path found");
        return path;
    }
    // Der Pfad wird zurückverfolgt
    path.0.push(conn.0.copy());
    path.1.push(conn.1.copy());
    let mut current = forward_visited.get(&conn).unwrap();
    while current != &start {
        path.0.insert(0, Point(current.0.0, current.0.1));
        path.1.insert(0, Point(current.1.0, current.1.1));
        current = forward_visited.get(current).unwrap();
    }
    path.0.insert(0, start.1.copy());
    path.1.insert(0, start.0.copy());
    current = backward_visited.get(&conn).unwrap();
    while current != &start {
        path.0.push(Point(current.0.0, current.0.1));
        path.1.push(Point(current.1.0, current.1.1));
        current = backward_visited.get(current).unwrap();
    }
    path
}

#[derive(Clone)]
#[repr(u8)]
enum Cell {
    NotVisited,      // Der Zustand wurde noch nicht besucht
    ForwardVisited,  // Der Zustand wurde von der Breitensuche vorwärts besucht
    BackwardVisited, // Der Zustand wurde von der Breitensuche rückwärts besucht
}

// Das Macro macht es einfacher, das Element für einen bestimmten zustand in dem Array zu finden
macro_rules! state_at {
    ($states:expr, $state:expr, $l1_width:expr, $l1_height:expr, $l2_width:expr) => {
        $states[($state.0.1 as usize * $l1_width as usize + $state.0.0 as usize)
            * $l2_width as usize
            * $l1_height as usize
            + $state.1.1 as usize * $l2_width as usize
            + $state.1.0 as usize]
    };
}

fn forward_array(
    l1: &Labyrinth,
    l2: &Labyrinth,
    current: &State,
    new_forward_queue: &mut Vec<State>,
    states: &mut Vec<Cell>,
    l1_width: i16,
    l1_height: i16,
    l2_width: i16,
) -> Option<State> {
    for inst in Instruction::iter() {
        let new_pos = State(
            l1.shift_point(&current.0, &inst),
            l2.shift_point(&current.1, &inst),
        );
        if &new_pos == current {
            continue;
        }
        match state_at!(states, new_pos, l1_width, l1_height, l2_width) {
            Cell::NotVisited => {
                // Der Zustand wurde noch nicht besucht und wird daher als vorwärts besucht markiert
                state_at!(states, new_pos, l1_width, l1_height, l2_width) = Cell::ForwardVisited;
                new_forward_queue.push(new_pos);
            }
            // Der Zustand wurde schon von der anderen 
            // Breitensuche besucht, d.h. ein Pfad ist gefunden
            Cell::BackwardVisited => return Some(new_pos),
            // Der Pfad wurde schon von der eigenen Breitensuche besucht,
            //  d.h. der Knoten wird nicht an die new_forward_queue angehängt
            Cell::ForwardVisited => continue,
        }
    }
    None
}

fn backward_array(
    l1: &Labyrinth,
    l2: &Labyrinth,
    current: &State,
    new_backward_queue: &mut Vec<State>,
    states: &mut Vec<Cell>,
    l1_width: i16,
    l1_height: i16,
    l2_width: i16,
) -> BackwardResult {
    let mut can_continue = true;
    for inst in Instruction::iter() {
        // Alle möglichen Punkte in Richtung inst werden gesucht
        let mut temp_queue: Vec<State> = vec![];
        if l1.is_wall_in_direction(&current.0, &inst.inv()) {
            temp_queue.push(State(
                current.0.copy(),
                l2.shift_point_without_end_fix(&current.1, &inst),
            ));
        }
        if l2.is_wall_in_direction(&current.1, &inst.inv()) {
            temp_queue.push(State(
                l1.shift_point_without_end_fix(&current.0, &inst),
                current.1.copy(),
            ));
        }
        temp_queue.push(State(
            l1.shift_point_without_end_fix(&current.0, &inst),
            l2.shift_point_without_end_fix(&current.1, &inst),
        ));
        for new_pos in temp_queue {
            if &new_pos == current {
                continue;
            }
            match state_at!(states, new_pos, l1_width, l1_height, l2_width) {
                Cell::NotVisited => {
                    if &State(
                        l1.shift_point(&new_pos.0, &inst.inv()),
                        l2.shift_point(&new_pos.1, &inst.inv()),
                    ) != current
                    {
                        continue;
                    }
                    state_at!(states, new_pos, l1_width, l1_height, l2_width) =
                        Cell::BackwardVisited;
                    if new_pos.0 == Point(0, 0) || new_pos.1 == Point(0, 0) {
                        // Anton oder Bea sind am Startfeld und könnten 
                        // aus jedem Loch gekommen sein
                        can_continue = false;
                    }
                    new_backward_queue.push(new_pos);
                }
                // Der Zustand wurde schon vorwärts besucht, 
                // d.h. ein Pfad wurde gefunden 
                Cell::ForwardVisited => return BackwardResult::Connection(new_pos),
                // Der Zustand wurde schon Rückwärts besucht, 
                // d.h. er wird nicht an die new_backward_queue angehängt
                Cell::BackwardVisited => continue,
            }
        }
    }
    if !can_continue {
        BackwardResult::CannotContinue
    } else {
        BackwardResult::None
    }
}
fn double_path_bidibfs_array(
    l1: &Labyrinth,
    l2: &Labyrinth,
    path: (Vec<Point>, Vec<Point>),
) -> (Vec<Point>, Vec<Point>) {
    assert!(l1.width == l2.width && l1.height == l2.height);
    let start = State(Point(0, 0), Point(0, 0));
    let end = State(
        Point(l1.width - 1, l1.height - 1),
        Point(l2.width - 1, l2.height - 1),
    );

    let total_states =
        (l1.width as usize) * (l1.height as usize) * (l2.width as usize) * (l2.height as usize);
    let mut states = vec![Cell::NotVisited; total_states];
    print!(
        "Creating state array of size {} (Size {} mb)\n",
        total_states,
        (total_states * size_of::<Cell>()) / 1024 / 1024
    );
    print!("Size of one Cell: {} bytes\n", std::mem::size_of::<Cell>());
    state_at!(states, start, l1.width, l1.height, l2.width) = Cell::ForwardVisited;
    state_at!(states, end, l1.width, l1.height, l2.width) = Cell::BackwardVisited;

    // Jeweils zwei Warteschlangen für jede Richtungen, zwischen denen hin und her gewechselt wird
    let mut forward_queue1: Vec<State> = vec![start.copy()];
    let mut backward_queue1: Vec<State> = vec![end.copy()];
    let mut forward_queue2: Vec<State> = vec![start.copy()];
    let mut backward_queue2: Vec<State> = vec![end.copy()];
    let mut forward_queue = &mut forward_queue1;
    let mut forward_queue_result = &mut forward_queue2;
    let mut backward_queue = &mut backward_queue1;
    let mut backward_queue_result = &mut backward_queue2;

    // Speichert, ob die Suche in Rückwärtsrichtung vorgeführt werden kann.
    let mut can_continue_backwards = true;

    let mut found_path = false;
    let mut conn = State(Point(0, 0), Point(0, 0));
    let mut depth = 0;
    while (forward_queue.len() > 0 && backward_queue.len() > 0) && !found_path {
        if depth % 100 == 0 {
            println!(
                "Depth: {}, Forward Stack {}, Backward Stack {}",
                depth,
                forward_queue.len(),
                backward_queue.len(),
            );
        }
        depth += 1;
        // Es wird ausgewählt, in welche Richtung der nächste Schritt gehen soll
        // wenn noch rückwärts weitergemacht werden kann, wird die Richtung mit 
        // der kürzeren Warteschlange gewählt
        if forward_queue.len() <= backward_queue.len() || !can_continue_backwards {
            // Die neue Warteschlange wird geleert
            forward_queue_result.clear();
            for current in &*forward_queue {
                // Jeder Knoten wird vorwärts expandiert
                let result = forward_array(
                    l1,
                    l2,
                    current,
                    forward_queue_result,
                    &mut states,
                    l1.width,
                    l1.height,
                    l2.width,
                );
                match result {
                    None => {} 
                    Some(state) => {
                        // Ein Pfad wurde gefunden
                        found_path = true;
                        conn = state;
                        continue;
                    }
                }
            }
            // Die Warteschlangen werden ausgetauscht
            let temp = forward_queue_result;
            forward_queue_result = forward_queue;
            forward_queue = temp;
        } else {
            // Die neue Warteschlange wird geleert
            backward_queue_result.clear();
            for current in &*backward_queue {
                // Jeder Knoten wird rückwärts expandiert
                let result = backward_array(
                    l1,
                    l2,
                    current,
                    backward_queue_result,
                    &mut states,
                    l1.width,
                    l1.height,
                    l2.width,
                );
                match result {
                    BackwardResult::None => {}
                    BackwardResult::Connection(state) => {
                        // Ein Pfad wurde gefunden
                        found_path = true;
                        conn = state;
                        break;
                    }
                    BackwardResult::CannotContinue => {
                        // Ein Punkt ist (0,0), d.h. die Suche 
                        // rückwärts wird nicht weitergeführt
                        can_continue_backwards = false;
                    }
                }
            }
            // Die Warteschlangen werden ausgetauscht
            let temp = backward_queue_result;
            backward_queue_result = backward_queue;
            backward_queue = temp;
        }
    }
    if !found_path {
        println!("No path found");
        return path;
    } else {
        println!("Found path at level {}", depth);
        path
    }
}

/// Die Heuristik für den A* Algorithmus
fn heuristic(state: &State, width: i16, height: i16) -> i16 {
    ((width - 1) - state.0.0 + (height - 1) - state.0.1)
        .max((width - 1) - state.1.0 + (height - 1) - state.1.1)
}

// Ein Zustand wird gemeinsam mit seinem Wert von g gespeichert
// Dadurch spart man sich eine separate HashMap
struct AStarState(State, i16);


// Der Wert von g wird für die Gleichheit nicht berücksichtigt
impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Der Wert von g wird beim Hashing nicht berücksichtigt
impl std::hash::Hash for AStarState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}


// Die Information, die sonst in der closed liste gespeichert wird
// wird in den Elementen der CameFrom HashMap gespeichert
struct CameFrom {
    state: State,
    is_closed: bool,
}

impl Eq for AStarState {}

fn double_path_a_star(
    l1: &Labyrinth,
    l2: &Labyrinth,
    mut path: (Vec<Point>, Vec<Point>),
) -> (Vec<Point>, Vec<Point>) {
    let start = State(Point(0, 0), Point(0, 0));

    let end = State(
        Point(l1.width - 1, l1.height - 1),
        Point(l2.width - 1, l2.height - 1),
    );
    // Diese PriorityQueue ist eine Max-Heap, daher werden die Prioritäten negiert,
    // um die Ordnung umzudrehen
    let mut open: PriorityQueue<AStarState, i16> = PriorityQueue::new();
    // Speichert, woher der Zustand erreicht wurde, und ob der Zustand teil der Closed Liste ist
    // d.h. wenn cameFrom[state].is_closed = true, dann ist state in der Closed Liste
    let mut cameFrom: HashMap<State, CameFrom> = HashMap::new();

    cameFrom.insert(
        start.copy(),
        CameFrom {
            state: start.copy(),
            is_closed: false,
        },
    );
    open.push(AStarState(start.copy(), 0), 0);

    while !open.is_empty() {
        let (current_a_star_state, _) = open.pop().unwrap();
        let current = current_a_star_state.0;
        let current_g = current_a_star_state.1;
        
        // Der Zustand ist "abgeschlossen"
        cameFrom.get_mut(&current).unwrap().is_closed = true;

        if current == end {
            // Ein Pfad ist gefunden
            break;
        }
        // Die neuen Kosten sind einfach zu berechnen, da jede Anweisung 
        // gleich viel "kostet"
        let new_cost = current_g + 1;
        for inst in Instruction::iter() {
            let new_pos = State(
                l1.shift_point(&current.0, &inst),
                l2.shift_point(&current.1, &inst),
            );
            if (new_pos == current) {
                continue;
            }

            // Wenn die Position schon in der closed_list enthalten ist, wird sie übersprungen
            match cameFrom.get_mut(&new_pos) {
                Some(came_from) => {
                    if came_from.is_closed {
                        // Die Position wurde schon besucht
                        continue;
                    }
                }
                None => {}
            }

            match open.get_mut(&AStarState(new_pos.copy(), 0)) {
                Some(mut new_AStarState) => {
                    // Der Zustand ist schon in der Open list
                    if (new_cost) >= new_AStarState.0.1 {
                        // Es ist schon ein besser Pfad zu new_pos bekannt
                        continue;
                    } else {
                        // Dieser Pfad zu new_pos ist besser, d.h. der Wert von g von new_pos
                        // wird aktualisiert
                        new_AStarState.0.1 = new_cost;
                    }
                }
                None => {
                    // Die Position ist noch nicht in der Open List
                    open.push(
                        AStarState(new_pos.copy(), new_cost),
                        -(new_cost + heuristic(&new_pos, l1.width, l1.height)),
                    );
                    cameFrom.insert(
                        new_pos.copy(),
                        CameFrom {
                            state: current.copy(),
                            is_closed: false,
                        },
                    );
                    continue;
                }
            }
            cameFrom.insert(
                new_pos.copy(),
                CameFrom {
                    state: current.copy(),
                    is_closed: false,
                },
            );
            // Die Priorität in der Open List wird aktualisiert
            open.change_priority(
                &AStarState(new_pos.copy(), 0),
                -(new_cost + heuristic(&new_pos, l1.width, l1.height)),
            );
        }
    }
    // Wenn die Position schon in der closed_list enthalten ist, wird sie übersprungen
    match cameFrom.get_mut(&end) {
        Some(_) => {}
        None => {
            println!("No path found");
            return path;
        }
    }

    // Der Pfad wird zurückverfolgt
    path.0.push(end.0.copy());
    path.1.push(end.1.copy());
    let mut current = &end.copy();
    while current != &start.copy() {
        current = &cameFrom.get(&current).unwrap().state;
        path.0.push(Point(current.0.0, current.0.1));
        path.1.push(Point(current.1.0, current.1.1));
    }
    path.0.reverse();
    path.1.reverse();

    path
}

fn main() {
    let number = Args::parse().file;
    let filename = "Examples/labyrinthe".to_string() + &number.to_string() + ".txt";
    let (labyrinth1, labyrinth2) = parse_file(&filename);
    let path: (Vec<Point>, Vec<Point>);

    // Array-Implementierung der Bidirektionalen Suche
    let start = Instant::now();
    double_path_bidibfs_array(&labyrinth1, &labyrinth2, (vec![], vec![]));
    let duration = start.elapsed();
    println!("Bidirectional Array BFS: {:?}", duration);

    // Bidirektionale Suche
    let start = Instant::now();
    let path2 = double_path_bidibfs(&labyrinth1, &labyrinth2, (vec![], vec![]));
    let duration = start.elapsed();
    println!("Bidirectional BFS: {:?}", duration);
    print_instruction_sequence(&instruction_sequence_from_path(
        &labyrinth1,
        &labyrinth2,
        &path2,
    ));
    labyrinth1
        .visualize_gradient(
            &path2.0,
            Args::parse().output_file.to_string() + &Args::parse().file.to_string() + "_1_bidi.png",
        )
        .unwrap();
    labyrinth2
        .visualize_gradient(
            &path2.1,
            Args::parse().output_file.to_string() + &Args::parse().file.to_string() + "_2_bidi.png",
        )
        .unwrap();
    println!("Path length: {}", path2.0.len());

    // Normale Breitensuche
    let start = Instant::now();
    path = double_path_bfs(&labyrinth1, &labyrinth2, (vec![], vec![]));
    let duration = start.elapsed();
    println!("BFS: {:?}", duration);
    print_instruction_sequence(&instruction_sequence_from_path(
        &labyrinth1,
        &labyrinth2,
        &path,
    ));
    labyrinth1
        .visualize_gradient(
            &path.0,
            Args::parse().output_file.to_string() + &Args::parse().file.to_string() + "_1.png",
        )
        .unwrap();
    labyrinth2
        .visualize_gradient(
            &path.1,
            Args::parse().output_file.to_string() + &Args::parse().file.to_string() + "_2.png",
        )
        .unwrap();
}
