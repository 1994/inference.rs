use infer_core::*;
use infer_ir::*;
use infer_scheduler::aggregate;

#[test]
fn page_cost_tracks_boundaries_of_complete_candidate_chunks() {
    let mut q = CostQuery::from_unit(
        ProgramId::new(1).unwrap(),
        BackendKind::Metal,
        ExecutionRole::Prefill,
        1,
        4,
        CostEstimate {
            gpu_us: 1,
            ..Default::default()
        },
    );
    q.page_growth = Some(PageGrowth {
        page_tokens: 4,
        allocated_pages: 1,
        bytes_per_page: 256,
        cow_tail: false,
    });
    q.logical_growth = Some(PageGrowth {
        page_tokens: 8,
        allocated_pages: 1,
        bytes_per_page: 0,
        cow_tail: false,
    });
    assert_eq!(aggregate(&[q]).unwrap().state_pages, 0);
    q.tokens = 6;
    q.context_tokens = 9;
    let cost = aggregate(&[q]).unwrap();
    assert_eq!(cost.state_pages, 2);
    assert_eq!(cost.state_bytes, 512);
    assert_eq!(cost.logical_pages, 1);
    q.page_growth.as_mut().unwrap().cow_tail = true;
    assert_eq!(aggregate(&[q]).unwrap().state_pages, 3);
    q.page_growth.as_mut().unwrap().page_tokens = 0;
    assert!(aggregate(&[q]).is_err());
}
