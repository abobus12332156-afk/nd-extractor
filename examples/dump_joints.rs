use nd_extractor::skeleton::Skeleton;
fn main() {
    let skeleton = Skeleton::parse_path(std::env::args().nth(1).unwrap()).unwrap();
    for b in 0..skeleton.bone_count() {
        let name = &skeleton.names[b];
        if !name.contains("twist") { continue; }
        let parent = skeleton.parents[b];
        let pname = if parent >= 0 { skeleton.names[parent as usize].clone() } else { "-1".to_string() };
        let ptarget = if parent >= 0 { skeleton.helper_target(parent as usize).map(|t| skeleton.names[t].clone()) } else { None };
        let target = skeleton.helper_target(b).map(|t| format!("{} ({})", t, skeleton.names[t]));
        println!("{} idx={} helper={} parent={} ({}) parent_target={:?} chain={} target={:?}", name, b, skeleton.helper_bones[b], parent, pname, ptarget, skeleton.chain_ids[b], target);
    }
}
