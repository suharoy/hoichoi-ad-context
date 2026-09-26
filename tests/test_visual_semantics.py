import copy

import numpy as np
import pytest

from src.context.visual_embeddings import normalize_vectors, context_vector
from scripts.embed_shots import load_shots
from scripts.enrich_visual_semantics import boundary_position, enrich_video


def shots():
    return [
        {"index": i + 1, "start": float(i), "end": float(i + 1),
         "representative_timestamp": i + 0.5}
        for i in range(4)
    ]


def test_normalization_and_invalid_vectors():
    vectors = normalize_vectors(np.array([[3., 4.], [0., 2.]]))
    np.testing.assert_allclose(vectors, [[.6, .8], [0, 1]])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
    for invalid in ([[0., 0.]], [[float('nan'), 1.]], [[float('inf'), 1.]]):
        with pytest.raises(ValueError):
            normalize_vectors(np.array(invalid))


def test_context_is_normalized_mean_with_available_shots():
    vectors = np.array([[1., 0.], [0., 1.]])
    np.testing.assert_allclose(context_vector(vectors), [2 ** -.5, 2 ** -.5])
    np.testing.assert_allclose(context_vector(vectors[:1]), [1, 0])
    assert context_vector(vectors[:0]) is None


def test_timestamp_mapping_and_edges():
    assert boundary_position(shots(), 1.) == 1
    assert boundary_position(shots(), 0.) == 0
    assert boundary_position(shots(), 4.) == 4
    with pytest.raises(ValueError):
        boundary_position(shots(), 1.5)
    bad = shots()
    bad[0]['end'] = .8
    with pytest.raises(ValueError):
        boundary_position(bad, 1.)


def test_csv_uses_seconds_not_frame_arithmetic(tmp_path):
    csv = tmp_path / 'shots.csv'
    csv.write_text(
        'Timecode List:,00:00:01.042\n'
        'Scene Number,Start Frame,Start Time (seconds),End Frame,End Time (seconds)\n'
        '1,1,0.000,25,1.042\n2,26,1.042,48,2.000\n', encoding='utf-8')
    result = load_shots(csv)
    assert result[1]['start'] == 1.042
    assert result[1]['representative_timestamp'] == 1.521
    assert boundary_position(result, 1.042) == 1


def test_features_context_and_prior_field_preservation():
    video = {'video': 'mandaar.mp4', 'candidates': [
        {'timestamp_seconds': 1., 'speech_safe': False, 'existing': {'nested': [1, 2]}},
        {'timestamp_seconds': 2., 'speech_safe': True, 'text_semantics': {'available': False}},
        {'timestamp_seconds': 0., 'speech_safe': True},
        {'timestamp_seconds': 4., 'speech_safe': True},
    ]}
    before = copy.deepcopy(video)
    vectors = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    result = enrich_video(video, shots(), vectors)
    assert video == before
    middle = result['candidates'][1]['visual_semantics']
    assert middle['previous_shot']['index'] == 2
    assert middle['next_shot']['index'] == 3
    assert middle['left_context_shots'] == [1, 2]
    assert middle['right_context_shots'] == [3, 4]
    assert middle['immediate_change'] == 1
    assert middle['context_change'] == pytest.approx(0, abs=1e-6)
    assert result['candidates'][0]['visual_semantics']['left_context_shots'] == [1]
    assert result['candidates'][2]['visual_semantics']['previous_shot'] is None
    assert result['candidates'][2]['visual_semantics']['context_change'] is None
    assert result['candidates'][3]['visual_semantics']['next_shot'] is None
    for c in result['candidates']:
        c.pop('visual_semantics')
    assert result == before
