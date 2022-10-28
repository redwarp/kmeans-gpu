use std::{
    cell::RefCell,
    collections::VecDeque,
    ops::{AddAssign, Deref},
    rc::Rc,
};

const MAX_DEPTH: u8 = 8;

fn get_color_index(color: &[u8; 4], level: u8) -> usize {
    let mut index: usize = 0;
    let mask = 0b10000000 >> level;
    if (color[0] & mask) > 0 {
        index |= 0b100;
    }
    if (color[1] & mask) > 0 {
        index |= 0b010;
    }
    if (color[2] & mask) > 0 {
        index |= 0b001;
    }

    index
}

pub(crate) struct ColorTree {
    nodes: Vec<Rc<RefCell<Node>>>,
}

impl ColorTree {
    #[must_use]
    pub fn new() -> Self {
        let root = Node::root(NodeId(0));

        let nodes = vec![Rc::new(RefCell::new(root))];

        Self { nodes }
    }

    pub fn add_color(&mut self, color: &[u8; 4]) {
        let mut level = 0;
        let mut root_id = NodeId(0);

        while level < MAX_DEPTH {
            let color_index = get_color_index(color, level);

            if self.nodes[*root_id].borrow().children[color_index].is_none() {
                let new_node_id = self.nodes.len();
                let child_node =
                    Node::with_parent(NodeId(new_node_id), root_id, color_index, level);
                {
                    let mut node = self.nodes[*root_id].borrow_mut();
                    node.children[color_index] = Some(child_node.node_id);
                    node.child_count += 1;
                }
                self.nodes.push(Rc::new(RefCell::new(child_node)));
            }
            root_id = self.nodes[*root_id].borrow().children[color_index].unwrap();
            level += 1;
        }

        self.nodes[*root_id].borrow_mut().accumulator += color;
    }

    pub fn reduce(&mut self, color_count: usize) -> Vec<[u8; 4]> {
        if color_count == 0 {
            return vec![];
        }

        let mut leaves: Vec<Rc<RefCell<Node>>> = self
            .nodes
            .iter()
            .filter(|node| node.borrow().accumulator.pixel_count > 0)
            .cloned()
            .collect();
        leaves.sort_by(|a, b| a.cmp(b).reverse());
        let mut leaves: VecDeque<_> = leaves.into_iter().collect();

        while leaves.len() > color_count {
            let node = leaves.pop_back().expect("Len is > 0");
            let color_index = node.borrow().color_index;
            let parent_id = node.borrow().parent;
            if let Some(parent_id) = parent_id {
                let parent = &self.nodes[*parent_id];
                if let Ok(position) = leaves.binary_search_by(|probe| parent.cmp(probe)) {
                    leaves.remove(position);
                }
                {
                    let mut parent = parent.borrow_mut();
                    parent.accumulator += node.borrow().accumulator;
                    parent.child_count -= 1;
                    parent.children[color_index] = None;
                    node.borrow_mut().parent = None;
                }

                if let Err(position) = leaves.binary_search_by(|probe| parent.cmp(probe)) {
                    leaves.insert(position, parent.clone());
                }
            }
        }
        self.nodes.clear();

        let mut palette: Vec<_> = leaves
            .into_iter()
            .map(|node| node.borrow().accumulator.output_color())
            .collect();
        palette.sort();
        palette.dedup();

        palette
    }
}

#[derive(Clone, Copy)]
struct ColorAccumulator {
    pixel_count: u64,
    r: u64,
    g: u64,
    b: u64,
}

impl ColorAccumulator {
    fn new() -> Self {
        Self {
            pixel_count: 0,
            r: 0,
            g: 0,
            b: 0,
        }
    }

    fn output_color(&self) -> [u8; 4] {
        [
            (self.r / self.pixel_count) as u8,
            (self.g / self.pixel_count) as u8,
            (self.b / self.pixel_count) as u8,
            255,
        ]
    }
}

impl AddAssign<&[u8; 4]> for ColorAccumulator {
    fn add_assign(&mut self, rhs: &[u8; 4]) {
        self.r += rhs[0] as u64;
        self.g += rhs[1] as u64;
        self.b += rhs[2] as u64;
        self.pixel_count += 1;
    }
}

impl AddAssign for ColorAccumulator {
    fn add_assign(&mut self, rhs: Self) {
        self.r += rhs.r;
        self.g += rhs.g;
        self.b += rhs.b;
        self.pixel_count += rhs.pixel_count;
    }
}

#[derive(Clone, Copy, PartialEq)]
struct NodeId(usize);

impl Deref for NodeId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Node {
    level: u32,
    node_id: NodeId,
    color_index: usize,
    parent: Option<NodeId>,
    children: [Option<NodeId>; 8],
    child_count: u32,
    accumulator: ColorAccumulator,
}

impl Node {
    fn root(node_id: NodeId) -> Self {
        Self {
            level: 0,
            node_id,
            color_index: 0,
            parent: None,
            children: [None; 8],
            child_count: 0,
            accumulator: ColorAccumulator::new(),
        }
    }

    fn with_parent(node_id: NodeId, parent_node_id: NodeId, color_index: usize, level: u8) -> Self {
        Self {
            level: level as u32,
            node_id,
            color_index,
            parent: Some(parent_node_id),
            children: [None; 8],
            child_count: 0,
            accumulator: ColorAccumulator::new(),
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.eq(other) {
            return Some(std::cmp::Ordering::Equal);
        }

        match self.child_count.partial_cmp(&other.child_count) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        };

        let ac = self.accumulator.pixel_count >> self.level;
        let bc = other.accumulator.pixel_count >> other.level;

        match ac.partial_cmp(&bc) {
            Some(std::cmp::Ordering::Equal) => self.node_id.partial_cmp(&other.node_id),
            ord => ord,
        }
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::ColorTree;

    #[test]
    fn test_add_color() {
        let mut tree = ColorTree::new();

        let pixels = [
            [9, 10, 20, 255],
            [16, 20, 31, 255],
            [21, 29, 40, 255],
            [23, 32, 56, 255],
            [25, 51, 45, 255],
            [30, 29, 57, 255],
            [32, 46, 55, 255],
            [36, 21, 39, 255],
            [37, 58, 94, 255],
            [37, 86, 46, 255],
            [52, 28, 39, 255],
            [57, 74, 80, 255],
            [60, 94, 139, 255],
            [64, 39, 81, 255],
            [65, 29, 49, 255],
            [70, 130, 50, 255],
            [77, 43, 50, 255],
            [79, 143, 186, 255],
            [87, 114, 119, 255],
            [96, 44, 44, 255],
            [115, 190, 211, 255],
            [117, 36, 56, 255],
            [117, 167, 67, 255],
            [122, 54, 123, 255],
            [122, 72, 65, 255],
            [129, 151, 150, 255],
            [136, 75, 43, 255],
            [162, 62, 140, 255],
            [164, 221, 219, 255],
            [165, 48, 48, 255],
            [168, 181, 178, 255],
            [168, 202, 88, 255],
            [173, 119, 87, 255],
            [190, 119, 43, 255],
            [192, 148, 115, 255],
            [198, 81, 151, 255],
            [199, 207, 204, 255],
            [207, 87, 60, 255],
            [208, 218, 145, 255],
            [215, 181, 148, 255],
            [218, 134, 62, 255],
            [222, 158, 65, 255],
            [223, 132, 165, 255],
            [231, 213, 179, 255],
            [232, 193, 112, 255],
            [235, 237, 233, 255],
        ];
        println!("Sorting {} colors", pixels.len());
        for pixel in &pixels {
            tree.add_color(pixel);
        }

        let palette = tree.reduce(8);

        assert_eq!(8, palette.len());
    }
}
