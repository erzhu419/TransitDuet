"""Replay the actual admission/command order, not sorted equal-time guesses."""


def credit_account(trace, manager):
    if manager.kind != 'candidate':
        return dict(passed=True, candidate=False, admissions=0)
    remaining = {}
    closed = set()
    issued = tightened = spent = unused = 0
    commands = []
    admissions = 0
    for event in trace['credit_ledger']:
        key = event['bus'], event['trip']
        if event['kind'] == 'admit':
            old, new = remaining.get(key), event['remaining_s']
            assert key not in closed and old == event['old_remaining_s']
            assert new >= 0 and (old is None or new < old)
            assert event['proposal_s'] <= event['time_s']
            assert new == event['proposed_s'] - event['spent_since_proposal_s']
            assert event['query_index'] == len(commands)
            assert event['cv2_delta'] <= 1e-12 and event['cost_delta'] <= 1e-9
            if old is None:
                issued += new
            else:
                tightened += old - new
            remaining[key] = new
            admissions += 1
        elif event['kind'] == 'command':
            query = trace['queries'][len(commands)]
            assert event['query_index'] == len(commands)
            assert (query['bus'],query['trip'],query['station'],query['time_s'],query['executed_request_s']) == (
                *key,event['station'],event['time_s'],event['action_s'])
            action = event['action_s']
            assert action in query['legal']
            if key in remaining:
                assert 0 <= action <= remaining[key]
                remaining[key] -= action
                spent += action
            commands.append(event)
        else:
            assert event['kind'] == 'return' and remaining.pop(key) == event['unused_s']
            unused += event['unused_s']
            closed.add(key)
    assert len(commands) == len(trace['queries'])
    assert not remaining and not manager.pending and issued == tightened + spent + unused
    assert admissions == len(manager.admissions)
    return dict(passed=True,candidate=True,admissions=admissions,commands=len(commands),
                issued_s=issued,tightened_s=tightened,spent_s=spent,unused_s=unused,
                no_quota_increase=True,no_command_edits=True)
