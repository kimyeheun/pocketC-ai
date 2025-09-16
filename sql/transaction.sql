SELECT * FROM pocketc.transactions;

-- 데이터 초기화 
truncate table transactions;

-- user_id 저장 확인
select count(distinct(user_id)) from transactions;
select * from transactions where user_id = 8;

-- user_id 별 카테고리 소비량 확인하기 
SELECT t.user_id, SUM(t.amount) as amount, s.sub_name FROM transactions as t
JOIN sub_categories as s ON s.sub_id = t.sub_id
WHERE user_id = 6
GROUP BY t.user_id, t.sub_id
ORDER BY t.user_id, amount DESC;
