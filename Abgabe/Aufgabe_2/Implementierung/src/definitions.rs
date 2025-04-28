use colorgrad::{Gradient, GradientBuilder}; // 0.17.1
use image::{ImageBuffer, Pixel, Rgba};
use imageproc::drawing::{draw_filled_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;
use std::cmp::PartialEq;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
// 0.17.1

#[derive(EnumIter, Clone, Copy)]
#[repr(u8)]
pub enum Instruction {
    UP,
    DOWN,
    LEFT,
    RIGHT,
}

impl Instruction {
    pub fn inv(&self) -> Instruction {
        // Invertiert eine Instruction
        match self {
            Instruction::UP => Instruction::DOWN,
            Instruction::DOWN => Instruction::UP,
            Instruction::LEFT => Instruction::RIGHT,
            Instruction::RIGHT => Instruction::LEFT,
        }
    }
}

#[derive(Eq, Hash, PartialEq, Debug)]
pub struct Point(pub i16, pub i16);
impl Point {
    pub fn shift(&self, direction: &Instruction) -> Point {
        // Bewegt den Punkt in die Richtung der Instruction
        match direction {
            Instruction::UP => Point(self.0, self.1 - 1),
            Instruction::DOWN => Point(self.0, self.1 + 1),
            Instruction::LEFT => Point(self.0 - 1, self.1),
            Instruction::RIGHT => Point(self.0 + 1, self.1),
        }
    }
    pub fn copy(&self) -> Point {
        Point(self.0, self.1)
    }
}

#[derive(Eq, Hash, PartialEq, Debug)]
pub struct State(pub Point, pub Point);

impl State {
    pub(crate) fn copy(&self) -> State {
        State(Point(self.0.0, self.0.1), Point(self.1.0, self.1.1))
    }
}

pub struct Labyrinth {
    pub(crate) width: i16,
    pub height: i16,
    pub vertical_walls: Vec<Vec<bool>>,
    pub horizontal_walls: Vec<Vec<bool>>,
    pub holes: Vec<Point>,
}

// Wird für die visualisierung benötigt
fn line_pixels((x0, y0): (f32, f32), (x1, y1): (f32, f32), thickness: u32) -> Vec<(u32, u32)> {
    // Cast to integers
    let x0 = x0 as i32;
    let y0 = y0 as i32;
    let x1 = x1 as i32;
    let y1 = y1 as i32;

    // Bresenham setup
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut x = x0;
    let mut y = y0;

    // Radius around center-line
    let r = (thickness as i32) / 2;

    let mut pts = Vec::new();
    while {
        // For each center pixel, emit its neighborhood
        for dy_off in -r..=r {
            for dx_off in -r..=r {
                // Optionally enforce circular cross-section:
                // if dx_off*dx_off + dy_off*dy_off <= r*r {
                let px = x + dx_off;
                let py = y + dy_off;
                if px >= 0 && py >= 0 {
                    pts.push((px as u32, py as u32));
                }
                // }
            }
        }

        // Advance Bresenham
        if x == x1 && y == y1 {
            false
        } else {
            let e2 = err * 2;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
            true
        }
    } {}

    pts
}
impl Labyrinth {
    /// Visualisiert das Labyrinth und einen Pfad
    pub fn visualize_gradient<P: AsRef<Path>>(
        &self,
        path: &Vec<Point>,
        filename: P,
    ) -> Result<(), Box<dyn Error>> {
        use image::{ImageBuffer, Rgba};
        use imageproc::drawing::{draw_filled_rect_mut, draw_line_segment_mut};
        use imageproc::rect::Rect;

        const CELL: u32 = 20;
        const THICK: u32 = 3;
        let width_px = (self.width as u32) * CELL + THICK;
        let height_px = (self.height as u32) * CELL + THICK;

        let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(width_px, height_px, Rgba([255, 255, 255, 255]));

        // Draw vertical walls
        for (row, row_walls) in self.vertical_walls.iter().enumerate() {
            for (col, &exists) in row_walls.iter().enumerate() {
                if exists {
                    let x = (col as u32 + 1) * CELL - THICK;
                    let y = row as u32 * CELL;
                    let rect = Rect::at(x as i32, y as i32).of_size(THICK * 2, CELL);
                    draw_filled_rect_mut(&mut img, rect, Rgba([0, 0, 0, 255]));
                }
            }
        }

        // Draw horizontal walls
        for (row, row_walls) in self.horizontal_walls.iter().enumerate() {
            for (col, &exists) in row_walls.iter().enumerate() {
                if exists {
                    let x = col as u32 * CELL;
                    let y = (row as u32 + 1) * CELL - THICK;
                    let rect = Rect::at(x as i32, y as i32).of_size(CELL, THICK * 2);
                    draw_filled_rect_mut(&mut img, rect, Rgba([0, 0, 0, 255]));
                }
            }
        }

        // Draw holes (in red), smaller
        for &Point(x, y) in &self.holes {
            let px = x as u32 * CELL + CELL / 4;
            let py = y as u32 * CELL + CELL / 4;
            let rect = Rect::at(px as i32, py as i32).of_size(CELL / 2, CELL / 2);
            draw_filled_rect_mut(&mut img, rect, Rgba([255, 0, 0, 255]));
        }

        // build a bright multi-stop gradient (blue → magenta → orange → yellow → green)
        let grad = GradientBuilder::new()
            .html_colors(&["#0066FF", "#CC00FF", "#FF6600", "#FFCC00", "#00CC66"])
            .domain(&[0.0, 1.0])
            .mode(colorgrad::BlendMode::Rgb)
            .build::<colorgrad::LinearGradient>()?;

        // Draw the path
        for (i, win) in path.windows(2).enumerate() {
            let &Point(x0, y0) = &win[0];
            let &Point(x1, y1) = &win[1];
            let cx0 = x0 as f32 * CELL as f32 + CELL as f32 / 2.0;
            let cy0 = y0 as f32 * CELL as f32 + CELL as f32 / 2.0;
            let cx1 = x1 as f32 * CELL as f32 + CELL as f32 / 2.0;
            let cy1 = y1 as f32 * CELL as f32 + CELL as f32 / 2.0;
            // get gradient color at this segment (normalized t)
            let t = i as f32 / (path.len() as f32);
            let [r, g, b, a] = grad.at(t).to_rgba8();
            // reduce base alpha so multiple segments still show overlap
            for (px, py) in line_pixels((cx0, cy0), (cx1, cy1), 10) {
                // add a color to the pixel
                let img_px = img.get_pixel_mut(px, py);
                *img_px = Rgba([r, g, b, a]);
            }
        }

        img.save(filename)?;
        Ok(())
    }

    
    /// Die folgenden Funktionen überprüfen, ob in einer bestimmten Richtung eine Wand ist
    /// (Oder ob in diese Richtung das Labyrinth zu ende ist)
    pub fn is_wall_below(&self, point: &Point) -> bool {
        if point.1 == self.height - 1 {
            return true;
        }
        self.horizontal_walls[point.1 as usize][point.0 as usize]
    }

    pub fn is_wall_right(&self, point: &Point) -> bool {
        if point.0 == self.width - 1 {
            return true;
        }
        self.vertical_walls[point.1 as usize][point.0 as usize]
    }

    pub fn is_wall_left(&self, point: &Point) -> bool {
        if point.0 == 0 {
            return true;
        }
        self.vertical_walls[point.1 as usize][point.0 as usize - 1]
    }

    pub fn is_wall_above(&self, point: &Point) -> bool {
        if point.1 == 0 {
            return true;
        }
        self.horizontal_walls[point.1 as usize - 1][point.0 as usize]
    }

    pub fn is_hole(&self, point: &Point) -> bool {
        for hole in &self.holes {
            if hole.0 == point.0 && hole.1 == point.1 {
                return true;
            }
        }
        false
    }

    pub fn is_wall_in_direction(&self, point: &Point, direction: &Instruction) -> bool {
        match direction {
            Instruction::UP => self.is_wall_above(point),
            Instruction::DOWN => self.is_wall_below(point),
            Instruction::LEFT => self.is_wall_left(point),
            Instruction::RIGHT => self.is_wall_right(point),
        }
    }
    
    /// Bewegt einen Punkt und beachtet dabei Wände, Löcher und Zielfelder
    pub fn shift_point(&self, point: &Point, direction: &Instruction) -> Point {
        if self.is_wall_in_direction(&point, direction) {
            return Point(point.0, point.1);
        }
        if point.0 == self.width - 1 && point.1 == self.height - 1 {
            return Point(point.0, point.1);
        }

        if self.is_hole(&point) {
            return Point(0, 0);
        }
        point.shift(direction)
    }
    
    /// Bewegt einen Punkt wie die shift_point Funktion, außer dass am Ende nicht 
    /// stillgestanden wird. Das wird für die Breitensuche rückwärts benötigt
    pub fn shift_point_without_end_fix(&self, point: &Point, direction: &Instruction) -> Point {
        if self.is_wall_in_direction(&point, direction) {
            return Point(point.0, point.1);
        }
        if self.is_hole(&point) {
            return Point(0, 0);
        }
        point.shift(direction)
    }
}

/// Liest eine Datei ein
pub fn parse_file(filename: &str) -> (Labyrinth, Labyrinth) {
    let path = Path::new(filename);
    let display = path.display();

    // Open the path in read-only mode (ignoring errors).
    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't Open {}: {}", display, why),
        Ok(file) => file,
    };

    // Read the file contents into a string.
    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", display, why),
        Ok(_) => println!("{}", display),
    }
    let mut lines = s.lines();
    let mut first_line = lines.next().unwrap().split_whitespace(); // The line has the format "width height"
    let width = i16::from_str(first_line.next().unwrap()).unwrap(); // n
    let height = i16::from_str(first_line.next().unwrap()).unwrap(); // m

    // m lines with n - 1 values (is there a wall to the right of the cell (1 or 0)
    let mut vertical_walls = vec![vec![false; (width - 1) as usize]; height as usize];
    for i in 0..height {
        let line = lines.next().unwrap();
        let mut values = line.split_whitespace();
        for j in 0..(width - 1) {
            if values.next().unwrap() == "1" {
                vertical_walls[i as usize][j as usize] = true;
            }
        }
    }
    // m - 1 lines with n values (is there a wall below the cell (1 or 0)
    let mut horizontal_walls = vec![vec![false; width as usize]; (height - 1) as usize];
    for i in 0..(height - 1) {
        let line = lines.next().unwrap();
        let mut values = line.split_whitespace();
        for j in 0..width {
            if values.next().unwrap() == "1" {
                horizontal_walls[i as usize][j as usize] = true;
            }
        }
    }
    // number of holes
    let num_holes = i16::from_str(lines.next().unwrap()).unwrap();
    // holes
    let mut holes = vec![];
    for _ in 0..num_holes {
        let line = lines.next().unwrap();
        let mut values = line.split_whitespace();
        let x = i16::from_str(values.next().unwrap()).unwrap();
        let y = i16::from_str(values.next().unwrap()).unwrap();
        holes.push(Point(x, y));
    }
    let labyrinth1 = Labyrinth {
        width,
        height,
        vertical_walls,
        horizontal_walls,
        holes,
    };

    // Now we have to read the second labyrinth
    // m lines with n - 1 values (is there a wall to the right of the cell (1 or 0)
    let mut vertical_walls = vec![vec![false; (width - 1) as usize]; height as usize];
    for i in 0..height {
        let line = lines.next().unwrap();
        let mut values = line.split_whitespace();
        for j in 0..(width - 1) {
            if values.next().unwrap() == "1" {
                vertical_walls[i as usize][j as usize] = true;
            }
        }
    }
    // m - 1 lines with n values (is there a wall below the cell (1 or 0)
    let mut horizontal_walls = vec![vec![false; width as usize]; (height - 1) as usize];
    for i in 0..(height - 1) {
        let line = lines.next().unwrap();
        let mut values = line.split_whitespace();
        for j in 0..width {
            if values.next().unwrap() == "1" {
                horizontal_walls[i as usize][j as usize] = true;
            }
        }
    }
    // number of holes
    let num_holes = i16::from_str(lines.next().unwrap()).unwrap();
    // holes
    let mut holes = vec![];
    for _ in 0..num_holes {
        let line = lines.next().unwrap();
        let mut values = line.split_whitespace();
        let x = i16::from_str(values.next().unwrap()).unwrap();
        let y = i16::from_str(values.next().unwrap()).unwrap();
        holes.push(Point(x, y));
    }
    let labyrinth2 = Labyrinth {
        width,
        height,
        vertical_walls,
        horizontal_walls,
        holes,
    };
    (labyrinth1, labyrinth2)
}


/// wird verwendet um die Anweisungssequenz auszugeben
pub fn instruction_from_points(
    l1: &Labyrinth,
    l2: &Labyrinth,
    p1: &Point,
    p2: &Point,
    p1_: &Point,
    p2_: &Point,
) -> Instruction {
    for inst in Instruction::iter() {
        let new_p1 = l1.shift_point(&p1, &inst);
        let new_p2 = l2.shift_point(&p2, &inst);
        if &new_p1 == p1_ && &new_p2 == p2_ {
            return inst;
        }
    }
    panic!("No instruction found for points p1:{:?}, p2: {:?}, p1_: {:?}, p2_: {:?}", p1, p2, p1_, p2_);
}


/// wird verwendet um die Anweisungssequenz auszugeben
pub fn instruction_sequence_from_path(
    l1: &Labyrinth,
    l2: &Labyrinth,
    path: &(Vec<Point>, Vec<Point>),
) -> Vec<Instruction> {
    if path.0.len() == 0 {
        return vec![];
    }
    let mut instructions = vec![];
    for i in 0..path.0.len() - 1 {
        instructions.push(instruction_from_points(
            l1,
            l2,
            &path.0[i],
            &path.1[i],
            &path.0[i + 1],
            &path.1[i + 1],
        ));
    }
    instructions
}


/// wird verwendet um die Anweisungssequenz auszugeben
pub fn print_instruction_sequence(instructions: &Vec<Instruction>) {
    // use unicode arrows
    let mut result = String::new();
    for inst in instructions {
        result.push(match inst {
            Instruction::UP => '↑',
            Instruction::DOWN => '↓',
            Instruction::LEFT => '←',
            Instruction::RIGHT => '→',
        });
    }
    println!("Länge: {}, {}",instructions.len(), result);
}
