-- mission_templates 예제 

INSERT INTO mission_templates
(name, locale, render_str, placeholders, dsl_skeletons, template_version)
VALUES
-- 1) 금지형: BAN_CATEGORY_DAILY
(
  'BAN_CATEGORY_DAILY',
  'ko-KR',
  '오늘 {label} 안 사기',
  JSON_OBJECT(
    'label', JSON_OBJECT('type','category_slug','required',TRUE)
  ),
  JSON_ARRAY(
    JSON_OBJECT(
      'id','ban_category_daily_v1',
      'evaluation','event_violation_only',
      'success_mode','eod_no_violation',
      'logic', JSON_OBJECT(
        'all', JSON_ARRAY(
          JSON_OBJECT('category_is','{label}')
        )
      )
    )
  ),
  1
),

-- 2) 예산형(일일 합계 상한): SPEND_CAP_DAILY
(
  'SPEND_CAP_DAILY',
  'ko-KR',
  '오늘 {label} {amount}원 이하',
  JSON_OBJECT(
    'label',  JSON_OBJECT('type','category_slug','required',TRUE),
    'amount', JSON_OBJECT('type','money','currency','KRW','min',0,'required',TRUE)
  ),
  JSON_ARRAY(
    JSON_OBJECT(
      'id','spend_cap_daily_v1',
      'evaluation','event_violation_only',
      'success_mode','eod_no_violation',
      'logic', JSON_OBJECT(
        'all', JSON_ARRAY(
          JSON_OBJECT('category_is','{label}'),
          /* 현재 이벤트까지 누적합(당일)이 {amount}를 초과하면 위반 */
          JSON_OBJECT('daily_sum_after_txn_gt','{amount}')
        )
      )
    )
  ),
  1
),

-- 3) 단일 결제 상한: PER_TXN_CAP_DAILY
(
  'PER_TXN_CAP_DAILY',
  'ko-KR',
  '{label} 1건 {per_txn}원 이하',
  JSON_OBJECT(
    'label',    JSON_OBJECT('type','category_slug','required',TRUE),
    'per_txn',  JSON_OBJECT('type','money','currency','KRW','min',0,'required',TRUE)
  ),
  JSON_ARRAY(
    JSON_OBJECT(
      'id','per_txn_cap_daily_v1',
      'evaluation','event_violation_only',
      'success_mode','eod_no_violation',
      'logic', JSON_OBJECT(
        'all', JSON_ARRAY(
          JSON_OBJECT('category_is','{label}'),
          JSON_OBJECT('amount_gt','{per_txn}')
        )
      )
    )
  ),
  1
),

-- 4) 시간제한 금지: TIME_WINDOW_BAN_DAILY
(
  'TIME_WINDOW_BAN_DAILY',
  'ko-KR',
  '{time_label} {label} 금지',
  JSON_OBJECT(
    'label',      JSON_OBJECT('type','category_slug','required',TRUE),
    'time_label', JSON_OBJECT('type','string','required',TRUE), -- UI 표시용: 예) "아침(09:00–12:00)"
    'time_window',JSON_OBJECT(
        'type','time_range','tz','Asia/Seoul','required',TRUE,
        'schema', JSON_OBJECT('start','HH:MM','end','HH:MM','allow_overnight',TRUE)
    )
  ),
  JSON_ARRAY(
    JSON_OBJECT(
      'id','time_window_ban_daily_v1',
      'evaluation','event_violation_only',
      'success_mode','eod_no_violation',
      'logic', JSON_OBJECT(
        'all', JSON_ARRAY(
          JSON_OBJECT('category_is','{label}'),
          JSON_OBJECT('time_in','{time_window}')  -- 22:00–06:00 같은 오버나이트 지원
        )
      )
    )
  ),
  1
),

-- 5) 빈도 제한(일일 횟수 상한): COUNT_CAP_DAILY
(
  'COUNT_CAP_DAILY',
  'ko-KR',
  '오늘 {label} {N}회 이하',
  JSON_OBJECT(
    'label', JSON_OBJECT('type','category_slug','required',TRUE),
    'N',     JSON_OBJECT('type','int','min',0,'max',20,'required',TRUE)
  ),
  JSON_ARRAY(
    JSON_OBJECT(
      'id','count_cap_daily_v1',
      'evaluation','event_violation_only',
      'success_mode','eod_no_violation',
      'logic', JSON_OBJECT(
        'all', JSON_ARRAY(
          JSON_OBJECT('category_is','{label}'),
          /* 현재 이벤트가 포함된 당일 카운트가 N을 초과하면 위반 */
          JSON_OBJECT('daily_count_after_txn_gt','{N}')
        )
      )
    )
  ),
  1
);
