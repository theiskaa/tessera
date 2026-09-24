//! Peak allocation of the Markdown path must not grow with document length. Native only: a
//! global allocator is per test binary, which is why this is its own file.

#![cfg(all(feature = "markdown", not(target_arch = "wasm32")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use tessera::{Config, Format, Kind, Query, Tessera};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded to `System` with the caller's arguments unchanged; the
// counters are updated with atomics and never influence the pointer returned.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), SeqCst) + layout.size();
        PEAK.fetch_max(live, SeqCst);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), SeqCst);
        // SAFETY: `ptr` came from `alloc` above, that is from `System`, with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const TEMPLATE: &str = "\n## Office {i}\n\n**Nino Beridze {i}**, Operations Lead  \nKavkaz Freight LLC  \n14 Rustaveli Avenue, Tbilisi 0108, Georgia  \n[nino{i}@kavkaz-freight.example](mailto:nino{i}@kavkaz-freight.example)\n\n- Phone: +44 20 7946 0958\n- Email: oliver{i}@hartley-sons.example\n\n| Name | Email |\n| --- | --- |\n| Ana {i} | ana{i}@kavkaz-freight.example |\n\n```\nbuild{i}@ci.example is inside a code block\n```\n\n> Regards,  \n> Ana Kapanadze\n";

const MIB: usize = 1024 * 1024;

fn document(target_bytes: usize) -> String {
    let mut out = String::from("# Directory\n");
    let mut i = 0;
    while out.len() < target_bytes {
        out.push_str(&TEMPLATE.replace("{i}", &i.to_string()));
        i += 1;
    }
    out
}

/// One `detect`'s allocation above what was live before it: at its peak, and retained by the
/// entities it returned.
struct Measured {
    peak: usize,
    retained: usize,
    entities: usize,
}

fn measure(t: &Tessera, text: &str) -> Measured {
    let query = Query {
        format: Format::Markdown(Default::default()),
        ..Query::default()
    };
    let live = LIVE.load(SeqCst);
    PEAK.store(live, SeqCst);
    let entities = t.detect(text, &query).unwrap();
    Measured {
        peak: PEAK.load(SeqCst) - live,
        retained: LIVE.load(SeqCst) - live,
        entities: entities.len(),
    }
}

#[test]
fn peak_allocation_does_not_grow_with_document_length() {
    let t = Tessera::load(
        &[],
        Config {
            kinds: Kind::Email | Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    let one = document(MIB);
    let five = document(5 * MIB);
    let a = measure(&t, &one);
    let b = measure(&t, &five);
    for (name, m) in [("1 MiB", &a), ("5 MiB", &b)] {
        println!(
            "{name}: peak {}, retained {}, working {}, {} entities",
            m.peak,
            m.retained,
            m.peak - m.retained,
            m.entities
        );
    }
    assert!(b.entities > 4 * a.entities);
    // A whole-document parse of 5 MiB peaks near 47 MB, so both bounds fail loudly if
    // segmenting is bypassed. The returned entities legitimately grow with the document
    // (about 190 bytes each), so the growth bound is on working memory, which excludes them;
    // what remains of its growth is the output vector's last reallocation. Measured: peaks of
    // 3.1 and 13.0 MB, working memory of 1.8 and 3.9 MB.
    assert!(
        b.peak <= 24 * MIB,
        "5 MiB document peaked at {} bytes",
        b.peak
    );
    let (working_one, working_five) = (a.peak - a.retained, b.peak - b.retained);
    assert!(
        working_five <= working_one + 4 * MIB,
        "working memory grew from {working_one} to {working_five} bytes between 1 MiB and 5 MiB"
    );
}
