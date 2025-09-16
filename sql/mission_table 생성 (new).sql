INSERT INTO mission_templates
(name, mission, placeholders, dsl, version)
VALUES
(
  'CATEGORY_BAN_DAILY',
  '오늘 {label} 안 사기',
  '{
    "label":      {"type":"string","desc":"미션 문구에 표시될 카테고리명","example":"담배"},
    "sub_id":     {"type":"int","desc":"대상 소분류 ID","example":101}
  }',
  '{
    "window": {
      "scope":"daily",
      "tz":"Asia/Seoul"
    },
    "conditions": [
      {
        "id":"ban_category_daily",
        "type":"count",
        "target":"transactions",
        "comparator":"==",
        "value":0,
        "params":{
          "sub_id":"{sub_id}"
        },
        "notes":"주어진 소분류의 일일 결제 건수가 0이어야 성공"
      }
    ]
  }',
  1
);

INSERT INTO mission_templates
(name, mission, placeholders, dsl, version)
VALUES
(
  'SPEND_CAP_DAILY',
  '오늘 {label} {amount}원 이하',
  '{
    "label":      {"type":"string","desc":"미션 문구에 표시될 카테고리명","example":"편의점"},
    "sub_id":     {"type":"int","desc":"대상 소분류 ID","example":102},
    "amount":     {"type":"int","desc":"일일 합계 상한(원)","example":15000}
  }',
  '{
    "window": {
      "scope":"daily",
      "tz":"Asia/Seoul"
    },
    "conditions": [
      {
        "id":"spend_cap_daily",
        "type":"spend",
        "target":"transactions",
        "comparator":"<=",
        "value":"{amount}",
        "params":{
          "sub_id":"{sub_id}"
        },
        "notes":"주어진 소분류의 일일 총 지출액이 amount 이하"
      }
    ]
  }',
  1
);

INSERT INTO mission_templates
(name, mission, placeholders, dsl, version)
VALUES
(
  'PER_TXN_DAILY',
  '{label} 1건 {per_txn}원 이하',
  '{
    "label":      {"type":"string","desc":"미션 문구에 표시될 카테고리명","example":"음료"},
    "sub_id":     {"type":"int","desc":"대상 소분류 ID","example":103},
    "per_txn":    {"type":"int","desc":"건별 금액 상한(원)","example":4000}
  }',
  '{
    "window": {
      "scope":"daily",
      "tz":"Asia/Seoul"
    },
    "conditions": [
      {
        "id":"per_txn_cap_daily",
        "type":"count",
        "target":"transactions",
        "comparator":"==",
        "value":0,
        "params":{
          "sub_id":"{sub_id}",
          "filter":{
            "amount":{">":"{per_txn}"}
          }
        },
        "notes":"건별 금액이 per_txn 를 초과하는 결제 건수가 0이어야 함"
      }
    ]
  }',
  1
);

INSERT INTO mission_templates
(name, mission, placeholders, dsl, version)
VALUES
(
  'TIME_BAN_DAILY',
  '{time_label} {label} 금지',
  '{
    "time_label": {"type":"string","desc":"문구용 시간대 레이블","example":"밤(22:00–06:00)"},
    "time_ranges":{"type":"array","desc":"시간대 리스트 [{start,end}]","example":[{"start":"22:00","end":"23:59"},{"start":"00:00","end":"06:00"}]},
    "label":      {"type":"string","desc":"미션 문구에 표시될 카테고리명","example":"야식 배달"},
    "sub_id":     {"type":"int","desc":"대상 소분류 ID","example":104}
  }',
  '{
    "window": {
      "scope":"daily",
      "tz":"Asia/Seoul",
      "time_of_day":"{time_ranges}"
    },
    "conditions": [
      {
        "id":"time_ban_daily",
        "type":"count",
        "target":"transactions",
        "comparator":"==",
        "value":0,
        "params":{
          "sub_id":"{sub_id}"
        },
        "notes":"지정된 시간대에 해당 소분류 결제가 0건이어야 함"
      }
    ]
  }',
  1
);

INSERT INTO mission_templates
(name, mission, placeholders, dsl, version)
VALUES
(
  'COUNT_CAP_DAILY',
  '오늘 {label} {N}회 이하',
  '{
    "label":      {"type":"string","desc":"미션 문구에 표시될 카테고리명","example":"커피"},
    "sub_id":     {"type":"int","desc":"대상 소분류 ID","example":105},
    "N":          {"type":"int","desc":"일일 허용 최대 횟수","example":2}
  }',
  '{
    "window": {
      "scope":"daily",
      "tz":"Asia/Seoul"
    },
    "conditions": [
      {
        "id":"count_cap_daily",
        "type":"count",
        "target":"transactions",
        "comparator":"<=",
        "value":"{N}",
        "params":{
          "sub_id":"{sub_id}"
        },
        "notes":"해당 소분류 결제 횟수가 N 이하"
      }
    ]
  }',
  1
);

