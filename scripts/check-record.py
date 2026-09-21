#!/usr/bin/env python3
"""Check recorded game consistency and report actual inference timing."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_k3.game import DIRECTIONS, SnakeGame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--max-mean-ms', type=float)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.record.read_text().splitlines()]
    settings = events[0]['settings']
    game = SnakeGame(settings['width'], settings['height'], settings['seed'], settings['initial_length'])
    times, rounds = [], 0
    for event in events[1:]:
        if event['type'] == 'frame':
            assert game.snapshot() == event['game'], 'Board/action mismatch'
            decision = event['decision']
            probabilities = decision['probabilities']
            assert set(probabilities) == set(DIRECTIONS)
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
            assert abs(sum(probabilities.values())-1) < .001
            assert decision['proposed'] == max(DIRECTIONS,key=probabilities.__getitem__)
            if events[0]['model'].get('policy') and not settings.get('unassisted',False):
                assert decision['executed'] in [m.direction for m in game.moves() if m.safe]
            game.step(decision['executed'])
            assert game.alive, 'Snake died'
            elapsed = decision['inference_ms']
            assert math.isfinite(elapsed) and elapsed > 0
            times.append(elapsed)
        elif event['type'] == 'round_end':
            assert game.snapshot() == event['game']
            assert game.won and game.alive
            rounds += 1
            game = SnakeGame(settings['width'],settings['height'],settings['seed']+rounds,settings['initial_length'])
    assert events[-1]['type'] == 'end', 'Incomplete recording'
    summary = events[-1]['summary']
    assert summary['steps'] == summary['inference_calls'] == len(times)
    assert events[-1]['game'] == game.snapshot()
    assert summary['deaths'] == 0
    if args.steps is not None:
        assert len(times) == args.steps, f'Expected {args.steps} steps, got {len(times)}'
    assert len(times) > 1, 'Need at least two steps for steady timing'
    steady = times[1:]
    result = {'steps':len(times),'completed_boards':rounds,'first_step_ms':times[0],
              'steady_mean_ms':statistics.mean(steady),'steady_median_ms':statistics.median(steady),
              'steady_p95_ms':sorted(steady)[int(.95*(len(steady)-1))],'steady_max_ms':max(steady),
              'deaths':summary['deaths'],'interventions':summary['interventions']}
    if args.max_mean_ms is not None:
        assert result['steady_mean_ms'] < args.max_mean_ms, result
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
