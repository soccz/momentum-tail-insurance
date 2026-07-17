# output/tables — 정본 표 데이터 안내

- 논문 표↔CSV 대응은 `paper/thesis_kr.md` 부록 D와 `docs/06_claims_map.md` 참조.
- **표 9(국제 경계조건)의 통합 소급본**: `table9_intl_boundary.csv` — 원천 CSV(fair_cross·p22·p23·us_leaderboard)와 원장 E번호를 행마다 명기. 미국 decile의 p값(0.67/0.75)과 한국 2×3의 n(291)은 원천 CSV에 없어 원장 E19·E22에서 소급했다.
- ⚠️ **`intl_cross.csv`는 미채택 구버전**(E20, vol-only 피처 파이프라인)이다. 미국 행이 +0.37(2×3 팩터)로 표 9의 미국 decile(−0.13)과 다른 것은 구성·피처 차이며, E20-정정에서 피처셋 오지정으로 판정되어 **논문에 사용되지 않았다**. 정본은 `fair_cross.csv`(E21 공정 재검정).
