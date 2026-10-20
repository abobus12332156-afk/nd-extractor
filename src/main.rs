use std::path::Path;
use std::process::ExitCode;

use nd_extractor::container::dsar::{DsarMetadataStats, DsarTableEntry};
use nd_extractor::container::pak::PakArchive;
use nd_extractor::container::psarc::{
    PsarcSummary, inspect_path_with_limit, inspect_path_with_metadata_stats,
};
use nd_extractor::glb::export_geometry_glb;

const DEFAULT_ENTRY_LIMIT: usize = 20;

fn main() -> ExitCode {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let arguments: Vec<_> = args.collect();
    match arguments.first().and_then(|argument| argument.to_str()) {
        Some("inspect-pak") => return inspect_pak(&program, &arguments[1..]),
        Some("extract-pak") => return extract_pak(&program, &arguments[1..]),
        Some("inspect-mesh") => return inspect_mesh(&program, &arguments[1..]),
        Some("export-mesh") => return export_mesh(&program, &arguments[1..]),
        Some("export-glb") => return export_glb(&program, &arguments[1..]),
        Some("inspect-textures") => return inspect_textures(&program, &arguments[1..]),
        Some("inspect-texture-graphs") => {
            return inspect_texture_graphs(&program, &arguments[1..]);
        }
        Some("inspect-dictionary-records") => {
            return inspect_dictionary_records(&program, &arguments[1..]);
        }
        Some("compare-texture-variants") => {
            return compare_texture_variants(&program, &arguments[1..]);
        }
        Some("extract-textures") => return extract_textures(&program, &arguments[1..]),
        Some("analyze-texture-payloads") => {
            return analyze_texture_payloads(&program, &arguments[1..]);
        }
        Some("analyze-texture-pitches") => {
            return analyze_texture_pitches(&program, &arguments[1..]);
        }
        _ => {}
    }
    if arguments.len() < 2 || arguments[0] != "inspect" {
        print_usage(&program);
        return ExitCode::from(2);
    }

    let archive_path = &arguments[1];
    let options = match parse_options(&arguments[2..]) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("error: {message}");
            print_usage(&program);
            return ExitCode::from(2);
        }
    };

    let path = Path::new(archive_path);
    let inspected = if options.metadata_stats {
        inspect_path_with_metadata_stats(path, options.limit)
            .map(|(summary, entries, stats)| (summary, entries, Some(stats)))
    } else {
        inspect_path_with_limit(path, options.limit)
            .map(|(summary, entries)| (summary, entries, None))
    };

    match inspected {
        Ok((summary, entries, stats)) => {
            print_summary(path, &summary);
            print_entries(&entries, summary.header.chunk_count);
            if let Some(stats) = stats {
                print_metadata_stats(&stats);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

struct InspectOptions {
    limit: usize,
    metadata_stats: bool,
}

fn print_usage(program: &std::ffi::OsStr) {
    eprintln!(
        "Usage:\n  {} inspect <archive.psarc> [--limit N] [--metadata-stats]\n  \
         {} inspect-pak <file.pak>\n  {} extract-pak <file.pak> --output <directory>\n  \
         {} inspect-mesh <file.pak> [--all] [--output <report.txt>]\n  \
         {} export-mesh <file.pak> [--all] --output <mesh.obj>\n  \
         {} export-glb <file.pak> --output <mesh.glb>\n  \
         {} inspect-textures <file.pak> [--all] [--dictionary <dict.pak>]\n  \
         {} inspect-texture-graphs <file.pak>\n  \
         {} inspect-dictionary-records <file.pak>\n  \
         {} compare-texture-variants <file.pak> [file.pak ...]\n  \
         {} extract-textures <file.pak> --output <directory> [--dictionary <dict.pak>]\n  \
         {} analyze-texture-payloads <file.pak> [--format FORMAT] [--limit N] (CSV to stdout)\n  \
         {} analyze-texture-pitches <file.pak> (CSV of candidate size models)",
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display(),
        Path::new(program).display()
    );
}

fn inspect_texture_graphs(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.len() != 1 {
        print_usage(program);
        return ExitCode::from(2);
    }
    match PakArchive::inspect_texture_graphs_path(Path::new(&arguments[0])) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_dictionary_records(
    program: &std::ffi::OsStr,
    arguments: &[std::ffi::OsString],
) -> ExitCode {
    if arguments.len() != 1 {
        print_usage(program);
        return ExitCode::from(2);
    }
    match PakArchive::inspect_texture_dictionary_records_path(Path::new(&arguments[0])) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn compare_texture_variants(
    program: &std::ffi::OsStr,
    arguments: &[std::ffi::OsString],
) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let paths = arguments.iter().map(Path::new);
    match PakArchive::compare_texture_variants_paths(paths) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_pak(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.len() != 1 {
        print_usage(program);
        return ExitCode::from(2);
    }
    let path = Path::new(&arguments[0]);
    match PakArchive::parse_path(path) {
        Ok(archive) => {
            println!("PAK inspection: {}", path.display());
            println!("Magic: {:#010x} (TLOU2 PAK variant)", archive.header.magic);
            println!("File size: {} bytes", archive.file_size);
            println!(
                "Header size: {}, pages: {}, pointer fixups: {}",
                archive.header.header_size,
                archive.pages.len(),
                archive.fixup_count
            );
            println!(
                "Login table: page {}, offset {:#x}",
                archive.header.login_page_index, archive.header.login_table_offset
            );
            println!("Pages:");
            for page in &archive.pages {
                println!(
                    "  [{}] class={} file_offset={:#x} stored_size={} resources={}",
                    page.index, page.page_class, page.offset, page.stored_size, page.resource_count
                );
            }
            println!("Resource type counts:");
            let mut type_counts = std::collections::BTreeMap::new();
            for resource in &archive.resources {
                *type_counts
                    .entry(resource.resource_type.as_str())
                    .or_insert(0usize) += 1;
            }
            for (resource_type, count) in type_counts {
                println!("  {resource_type}: {count}");
            }
            println!(
                "Resource inventory: {} entries; payload extents and decompression are not implemented",
                archive.resources.len()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn extract_pak(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.len() != 3 || arguments[1] != "--output" {
        print_usage(program);
        return ExitCode::from(2);
    }
    let source_path = Path::new(&arguments[0]);
    let output_path = Path::new(&arguments[2]);
    match PakArchive::parse_path(source_path) {
        Ok(archive) => match archive.extract_pages_path(source_path, output_path) {
            Ok(()) => {
                println!(
                    "Copied {} stored page segments and indexed {} resource records to {}",
                    archive.pages.len(),
                    archive.resources.len(),
                    output_path.display()
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_mesh(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let path = Path::new(&arguments[0]);
    let mut all = false;
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] == "--all" && !all {
            all = true;
            index += 1;
        } else if arguments[index] == "--output" && output.is_none() {
            let Some(path) = arguments.get(index + 1) else {
                eprintln!("error: --output requires a report path");
                return ExitCode::from(2);
            };
            output = Some(Path::new(path).to_owned());
            index += 2;
        } else {
            print_usage(program);
            return ExitCode::from(2);
        }
    }
    let report_result = if all {
        PakArchive::inspect_geometry_path(path)
            .map(|inventory| PakArchive::render_geometry_inventory(&inventory))
    } else {
        PakArchive::inspect_mesh_path(path).map(|mesh| PakArchive::render_mesh_report(&mesh.report))
    };
    match report_result {
        Ok(report) => {
            if let Some(output) = output {
                match std::fs::write(&output, report) {
                    Ok(()) => println!(
                        "Validated {} diagnostic report: {}",
                        if all { "geometry inventory" } else { "mesh" },
                        output.display()
                    ),
                    Err(error) => {
                        eprintln!("error: cannot write {}: {error}", output.display());
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                print!("{report}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn export_mesh(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let source_path = Path::new(&arguments[0]);
    let mut all = false;
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] == "--all" && !all {
            all = true;
            index += 1;
        } else if arguments[index] == "--output" && output.is_none() {
            let Some(path) = arguments.get(index + 1) else {
                eprintln!("error: --output requires an OBJ path");
                return ExitCode::from(2);
            };
            output = Some(Path::new(path).to_owned());
            index += 2;
        } else {
            print_usage(program);
            return ExitCode::from(2);
        }
    }
    let Some(output_path) = output else {
        eprintln!("error: export-mesh requires --output <mesh.obj>");
        return ExitCode::from(2);
    };
    if all {
        match PakArchive::inspect_geometry_path(source_path) {
            Ok(inventory) => match inventory.export_supported_obj(&output_path) {
                Ok(()) => {
                    let vertex_count: usize = inventory
                        .decoded_meshes
                        .iter()
                        .map(|mesh| mesh.vertices.len())
                        .sum();
                    let triangle_count: usize = inventory
                        .decoded_meshes
                        .iter()
                        .map(|mesh| mesh.indices.len() / 3)
                        .sum();
                    let unsupported = inventory
                        .submeshes
                        .iter()
                        .filter(|submesh| !submesh.supported)
                        .count();
                    println!(
                        "Exported {} supported submeshes: {} vertices, {} triangles; {} unsupported -> {}",
                        inventory.decoded_meshes.len(),
                        vertex_count,
                        triangle_count,
                        unsupported,
                        output_path.display()
                    );
                    for submesh in inventory.submeshes.iter().filter(|entry| !entry.supported) {
                        println!(
                            "  omitted submesh {} ({}): {}",
                            submesh.index,
                            submesh.position_format,
                            submesh.issue.as_deref().unwrap_or("unsupported")
                        );
                    }
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            },
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    } else {
        match PakArchive::inspect_mesh_path(source_path) {
            Ok(mesh) => match mesh.export_obj(&output_path) {
                Ok(()) => {
                    println!(
                        "Exported validated OBJ: {} vertices, {} triangles -> {}",
                        mesh.vertices.len(),
                        mesh.indices.len() / 3,
                        output_path.display()
                    );
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::FAILURE
                }
            },
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn inspect_textures(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let source_path = Path::new(&arguments[0]);
    let mut all = false;
    let mut dictionary = None;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] == "--all" && !all {
            all = true;
            index += 1;
        } else if arguments[index] == "--dictionary" && dictionary.is_none() {
            let Some(path) = arguments.get(index + 1) else {
                eprintln!("error: --dictionary requires a PAK path");
                return ExitCode::from(2);
            };
            dictionary = Some(Path::new(path).to_owned());
            index += 2;
        } else {
            print_usage(program);
            return ExitCode::from(2);
        }
    }
    match PakArchive::inspect_textures_path(source_path, dictionary.as_deref()) {
        Ok(inventory) => {
            if all {
                print!("{}", inventory.detailed_report());
            } else {
                print!("{}", inventory.summary());
                if let Some(first) = inventory.descriptors.first() {
                    println!(
                        "Example: resource={} hash={:#018x} {} {}x{} mips={} payload={} status={}",
                        first.descriptor.resource_index,
                        first.descriptor.hash,
                        first.descriptor.format_name,
                        first.descriptor.width,
                        first.descriptor.height,
                        first.descriptor.mip_count,
                        first.payload.as_ref().map_or_else(
                            || "unresolved".to_owned(),
                            |payload| format!(
                                "{}@{:#x} ({} bytes)",
                                payload.source_path.display(),
                                payload.source_data_offset,
                                payload.source_data_size
                            )
                        ),
                        first.status
                    );
                }
                println!("Use --all for per-descriptor metadata.");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn extract_textures(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let source_path = Path::new(&arguments[0]);
    let mut output = None;
    let mut dictionary = None;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] == "--output" && output.is_none() {
            let Some(path) = arguments.get(index + 1) else {
                eprintln!("error: --output requires a directory path");
                return ExitCode::from(2);
            };
            output = Some(Path::new(path).to_owned());
            index += 2;
        } else if arguments[index] == "--dictionary" && dictionary.is_none() {
            let Some(path) = arguments.get(index + 1) else {
                eprintln!("error: --dictionary requires a PAK path");
                return ExitCode::from(2);
            };
            dictionary = Some(Path::new(path).to_owned());
            index += 2;
        } else {
            print_usage(program);
            return ExitCode::from(2);
        }
    }
    let Some(output) = output else {
        eprintln!("error: extract-textures requires --output <directory>");
        return ExitCode::from(2);
    };
    match PakArchive::inspect_textures_path(source_path, dictionary.as_deref()) {
        Ok(inventory) => match inventory.extract_raw_payloads(&output) {
            Ok(extracted) => {
                let unresolved = inventory
                    .descriptors
                    .iter()
                    .filter(|entry| entry.payload.is_none())
                    .count();
                println!(
                    "Extracted {extracted}/{} raw GPU payloads to {}; PNG decoding is unsupported because the payload layout is unverified; {} unresolved",
                    inventory.descriptors.len(),
                    output.display(),
                    unresolved
                );
                println!(
                    "Manifest: {}",
                    output.join("texture-manifest.csv").display()
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn analyze_texture_payloads(
    program: &std::ffi::OsStr,
    arguments: &[std::ffi::OsString],
) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let source_path = Path::new(&arguments[0]);
    let mut format = None;
    let mut limit = 256usize;
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] == "--format" && format.is_none() {
            let Some(value) = arguments.get(index + 1).and_then(|value| value.to_str()) else {
                eprintln!("error: --format requires a UTF-8 format name");
                return ExitCode::from(2);
            };
            format = Some(value.to_owned());
            index += 2;
        } else if arguments[index] == "--limit" {
            let Some(value) = arguments.get(index + 1).and_then(|value| value.to_str()) else {
                eprintln!("error: --limit requires a positive integer");
                return ExitCode::from(2);
            };
            match value.parse::<usize>() {
                Ok(parsed) if parsed > 0 => limit = parsed,
                _ => {
                    eprintln!("error: --limit requires a positive integer");
                    return ExitCode::from(2);
                }
            }
            index += 2;
        } else {
            print_usage(program);
            return ExitCode::from(2);
        }
    }
    match PakArchive::analyze_texture_payloads_path(source_path, format.as_deref(), limit) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn analyze_texture_pitches(
    program: &std::ffi::OsStr,
    arguments: &[std::ffi::OsString],
) -> ExitCode {
    if arguments.len() != 1 {
        print_usage(program);
        return ExitCode::from(2);
    }
    match PakArchive::analyze_texture_pitch_models_path(Path::new(&arguments[0])) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_options(arguments: &[std::ffi::OsString]) -> Result<InspectOptions, String> {
    let mut options = InspectOptions {
        limit: DEFAULT_ENTRY_LIMIT,
        metadata_stats: false,
    };
    let mut index = 0;
    let mut limit_seen = false;

    while index < arguments.len() {
        if arguments[index] == "--metadata-stats" && !options.metadata_stats {
            options.metadata_stats = true;
            index += 1;
        } else if arguments[index] == "--limit" && !limit_seen {
            let value = arguments
                .get(index + 1)
                .and_then(|value| value.to_str())
                .ok_or_else(|| "--limit requires a non-negative integer".to_owned())?;
            options.limit = value
                .parse()
                .map_err(|_| "--limit must be a non-negative integer".to_owned())?;
            limit_seen = true;
            index += 2;
        } else {
            return Err("expected optional --limit N and --metadata-stats options".to_owned());
        }
    }

    Ok(options)
}

fn print_summary(path: &Path, summary: &PsarcSummary) {
    println!("Archive inspection: {}", path.display());
    println!("Detected container: DSAR (outer .psarc file)");
    println!("Magic: {}", String::from_utf8_lossy(&summary.header.magic));
    println!(
        "Version: {:#010x}{}",
        summary.header.version,
        if summary.header.version == 0x0001_0003 {
            " (observed TLOU2 PC version)"
        } else {
            ""
        }
    );
    println!("Archive size: {} bytes", summary.archive_size);
    println!("Chunk count: {}", summary.header.chunk_count);
    println!(
        "Chunk table: {:#x}..{:#x} ({} bytes; range validated)",
        summary.chunk_table.start,
        summary.chunk_table.end,
        summary.chunk_table.end - summary.chunk_table.start
    );
    println!(
        "Compressed payload range: {:#x}..{:#x} ({} bytes)",
        summary.compressed_payload.start,
        summary.compressed_payload.end,
        summary.compressed_payload.end - summary.compressed_payload.start
    );
    println!(
        "Declared decompressed size: {} bytes",
        summary.header.decompressed_size
    );
    println!(
        "Header padding marker: {}",
        String::from_utf8_lossy(&summary.header.padding_marker)
    );
    println!("Header inspection: supported for this DSAR version");
    println!("Entry byte at +0x18 meaning: unknown; reported as a raw byte value");
    println!("Inner PSARC directory and entry names: not inspected");
    println!("Extraction/decompression: unsupported");
}

fn print_metadata_stats(stats: &DsarMetadataStats) {
    println!(
        "Metadata diagnostics: validated {} table entries; payload was not read",
        stats.entries_validated
    );
    println!("Raw byte +0x18 value counts (meaning unknown):");
    for (value, count) in &stats.raw_byte_24_counts {
        println!("  {value:#04x}: {count}");
    }
    println!("Raw bytes +0x18..+0x1f sequence counts (meaning unknown):");
    for (bytes, count) in &stats.raw_bytes_24_to_31_counts {
        println!("  {}: {count}", hex_bytes(bytes));
    }
    if stats.untracked_raw_sequence_count > 0 {
        println!(
            "  additional raw sequences omitted after safety cap: {} occurrences",
            stats.untracked_raw_sequence_count
        );
    }
    println!("Records with compressed-size field equal to zero, grouped by raw byte:");
    if stats.zero_compressed_size_by_byte_24.is_empty() {
        println!("  none");
    } else {
        for (value, count) in &stats.zero_compressed_size_by_byte_24 {
            println!("  {value:#04x}: {count}");
        }
    }
}

fn print_entries(entries: &[DsarTableEntry], total_count: u32) {
    println!(
        "Table entries: displaying {} of {}",
        entries.len(),
        total_count
    );
    for entry in entries {
        println!(
            "  [{}] decompressed_offset={:#x}, compressed_offset={:#x}, \
             decompressed_size={}, compressed_size={}",
            entry.index,
            entry.decompressed_offset,
            entry.compressed_offset,
            entry.decompressed_size,
            entry.compressed_size
        );
        println!(
            "      range checks: decompressed offset+size within declared size; \
             compressed offset+size within archive"
        );
        println!(
            "      raw tag={:#04x} (meaning unknown), trailing bytes={}",
            entry.tag,
            hex_bytes(&entry.trailing_bytes)
        );
    }
    println!("Note: Only displayed records were parsed and validated.");
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn export_glb(program: &std::ffi::OsStr, arguments: &[std::ffi::OsString]) -> ExitCode {
    if arguments.is_empty() {
        print_usage(program);
        return ExitCode::from(2);
    }
    let source = Path::new(&arguments[0]);
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        if (arguments[index] == "--output" || arguments[index] == "-o") && output.is_none() {
            let Some(path) = arguments.get(index + 1) else {
                eprintln!("error: export-glb requires a path after --output/-o");
                return ExitCode::from(2);
            };
            output = Some(Path::new(path).to_owned());
            index += 2;
        } else {
            print_usage(program);
            return ExitCode::from(2);
        }
    }
    let Some(output) = output else {
        eprintln!("error: export-glb requires --output <mesh.glb>");
        return ExitCode::from(2);
    };
    match PakArchive::inspect_geometry_path(source) {
        Ok(inventory) => match export_geometry_glb(&inventory, &output) {
            Ok(summary) => {
                println!(
                    "Exported {} LOD0 static submeshes: {} vertices, {} triangles -> {}",
                    summary.mesh_count,
                    summary.vertex_count,
                    summary.triangle_count,
                    output.display()
                );
                println!(
                    "Normals, UVs, texture images, and skinning were not exported; materials are placeholders."
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
