"""
Analyze processed video results to understand what happened in the video
"""
import json
import sys
import codecs
from collections import Counter, defaultdict
from typing import Dict, List, Any

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def load_results(json_path: str) -> Dict[str, Any]:
    """Load results JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_detections(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze YOLO detections to understand video content

    Returns comprehensive analysis of what's in the video
    """
    detections = results.get('detections', [])

    # Track all objects across all frames
    all_objects = []
    objects_per_frame = []
    timeline_events = []

    # Track object appearances and positions
    object_timeline = defaultdict(list)  # {class_name: [(timestamp, bbox, confidence)]}

    for frame in detections:
        timestamp = frame['timestamp']
        frame_objects = frame.get('detections', [])
        objects_per_frame.append(len(frame_objects))

        frame_summary = {
            'timestamp': timestamp,
            'objects': {}
        }

        for obj in frame_objects:
            class_name = obj['class_name']
            confidence = obj['confidence']
            bbox = obj['bbox']

            all_objects.append(class_name)

            # Track object timeline
            object_timeline[class_name].append({
                'timestamp': timestamp,
                'bbox': bbox,
                'confidence': confidence
            })

            # Count objects per frame
            if class_name not in frame_summary['objects']:
                frame_summary['objects'][class_name] = 0
            frame_summary['objects'][class_name] += 1

        if frame_summary['objects']:
            timeline_events.append(frame_summary)

    # Calculate statistics
    object_counts = Counter(all_objects)
    unique_objects = list(object_counts.keys())

    # Find when objects appear/disappear
    object_appearances = {}
    for obj_class, appearances in object_timeline.items():
        first_seen = min(a['timestamp'] for a in appearances)
        last_seen = max(a['timestamp'] for a in appearances)
        avg_confidence = sum(a['confidence'] for a in appearances) / len(appearances)

        object_appearances[obj_class] = {
            'first_seen': first_seen,
            'last_seen': last_seen,
            'duration': last_seen - first_seen,
            'total_frames': len(appearances),
            'avg_confidence': avg_confidence,
            'appearances': appearances
        }

    # Detect interesting moments (significant changes in objects)
    interesting_moments = []
    prev_objects = set()
    for event in timeline_events:
        current_objects = set(event['objects'].keys())

        # New objects appeared
        new_objects = current_objects - prev_objects
        if new_objects:
            interesting_moments.append({
                'timestamp': event['timestamp'],
                'type': 'new_objects',
                'objects': list(new_objects),
                'description': f"New objects appeared: {', '.join(new_objects)}"
            })

        # Objects disappeared
        disappeared = prev_objects - current_objects
        if disappeared:
            interesting_moments.append({
                'timestamp': event['timestamp'],
                'type': 'objects_disappeared',
                'objects': list(disappeared),
                'description': f"Objects disappeared: {', '.join(disappeared)}"
            })

        prev_objects = current_objects

    return {
        'summary': {
            'total_frames': len(detections),
            'total_objects_detected': sum(objects_per_frame),
            'avg_objects_per_frame': sum(objects_per_frame) / len(objects_per_frame) if objects_per_frame else 0,
            'unique_object_types': len(unique_objects),
            'object_types': unique_objects
        },
        'object_counts': dict(object_counts),
        'object_timeline': object_appearances,
        'interesting_moments': interesting_moments,
        'timeline_events': timeline_events[:10]  # First 10 frames as sample
    }


def analyze_transcription(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Whisper transcription"""
    transcription = results.get('transcription', {})

    if 'error' in transcription:
        return {
            'status': 'error',
            'error': transcription['error']
        }

    segments = transcription.get('transcription', [])

    if not segments:
        return {
            'status': 'no_audio',
            'message': 'No speech detected in video'
        }

    # Extract all text
    full_text = ' '.join([seg['text'] for seg in segments])

    # Calculate stats
    total_duration = segments[-1]['end'] if segments else 0
    avg_confidence = sum(seg['confidence'] for seg in segments) / len(segments) if segments else 0

    # Find key phrases (simple word frequency)
    words = full_text.lower().split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(10)

    return {
        'status': 'success',
        'summary': {
            'language': transcription.get('language', 'unknown'),
            'total_segments': len(segments),
            'total_duration': total_duration,
            'avg_confidence': avg_confidence,
            'word_count': len(words)
        },
        'full_transcript': full_text,
        'segments': segments[:5],  # First 5 segments
        'common_words': common_words
    }


def generate_video_understanding(results: Dict[str, Any]) -> str:
    """
    Generate human-readable understanding of video content
    """
    detection_analysis = analyze_detections(results)
    transcription_analysis = analyze_transcription(results)

    # Build understanding report
    report = []
    report.append("="*80)
    report.append("VIDEO UNDERSTANDING REPORT")
    report.append("="*80)
    report.append("")

    # Video metadata
    report.append(f"Video: {results['video']}")
    report.append(f"Processed: {results['processed_at']}")
    report.append(f"Processing Time: {results['processing_time']:.2f}s")
    report.append("")

    # Visual Analysis
    report.append("="*80)
    report.append("VISUAL ANALYSIS (What's in the video)")
    report.append("="*80)
    report.append("")

    summary = detection_analysis['summary']
    report.append(f"📊 Summary:")
    report.append(f"  - Total frames analyzed: {summary['total_frames']}")
    report.append(f"  - Total objects detected: {summary['total_objects_detected']}")
    report.append(f"  - Average objects per frame: {summary['avg_objects_per_frame']:.1f}")
    report.append(f"  - Types of objects found: {summary['unique_object_types']}")
    report.append("")

    report.append(f"🎯 Objects Detected:")
    for obj, count in sorted(detection_analysis['object_counts'].items(), key=lambda x: x[1], reverse=True):
        report.append(f"  - {obj}: {count} times")
    report.append("")

    report.append(f"⏱️  Object Timeline:")
    for obj, info in sorted(detection_analysis['object_timeline'].items(),
                            key=lambda x: x[1]['duration'], reverse=True):
        report.append(f"  - {obj}:")
        report.append(f"      First seen: {info['first_seen']:.1f}s")
        report.append(f"      Last seen: {info['last_seen']:.1f}s")
        report.append(f"      Duration: {info['duration']:.1f}s")
        report.append(f"      Appearances: {info['total_frames']} frames")
        report.append(f"      Avg confidence: {info['avg_confidence']:.2f}")
    report.append("")

    if detection_analysis['interesting_moments']:
        report.append(f"🎬 Interesting Moments (Scene Changes):")
        for moment in detection_analysis['interesting_moments'][:10]:
            report.append(f"  - {moment['timestamp']:.1f}s: {moment['description']}")
        report.append("")

    # Audio Analysis
    report.append("="*80)
    report.append("AUDIO ANALYSIS (What's being said)")
    report.append("="*80)
    report.append("")

    if transcription_analysis['status'] == 'success':
        trans_summary = transcription_analysis['summary']
        report.append(f"🎤 Summary:")
        report.append(f"  - Language: {trans_summary['language']}")
        report.append(f"  - Total segments: {trans_summary['total_segments']}")
        report.append(f"  - Duration: {trans_summary['total_duration']:.1f}s")
        report.append(f"  - Word count: {trans_summary['word_count']}")
        report.append(f"  - Avg confidence: {trans_summary['avg_confidence']:.2f}")
        report.append("")

        report.append(f"💬 Transcript:")
        report.append(f"  {transcription_analysis['full_transcript'][:500]}...")
        report.append("")

        report.append(f"🔑 Most Common Words:")
        for word, count in transcription_analysis['common_words']:
            if len(word) > 3:  # Skip short words
                report.append(f"  - {word}: {count} times")
        report.append("")
    else:
        report.append(f"⚠️  Audio transcription status: {transcription_analysis.get('status', 'unknown')}")
        if 'error' in transcription_analysis:
            report.append(f"   Error: {transcription_analysis['error']}")
        report.append("")

    # Combined Understanding
    report.append("="*80)
    report.append("COMBINED UNDERSTANDING")
    report.append("="*80)
    report.append("")

    # Generate scene description
    report.append("📝 What's happening in this video:")
    report.append("")

    # Describe based on objects
    object_counts_sorted = sorted(detection_analysis['object_counts'].items(), key=lambda x: x[1], reverse=True)
    main_objects = [obj for obj, count in object_counts_sorted[:3]]
    if 'person' in main_objects:
        person_count = detection_analysis['object_counts']['person']
        person_info = detection_analysis['object_timeline']['person']
        report.append(f"  • There are people in the video ({person_count} total detections)")
        report.append(f"    - Present from {person_info['first_seen']:.1f}s to {person_info['last_seen']:.1f}s")

    other_objects = [obj for obj in main_objects if obj != 'person']
    if other_objects:
        report.append(f"  • Main objects: {', '.join(other_objects)}")

    report.append("")

    # Describe activity level
    avg_objects = summary['avg_objects_per_frame']
    if avg_objects < 2:
        report.append("  • Scene complexity: Simple (few objects)")
    elif avg_objects < 5:
        report.append("  • Scene complexity: Moderate")
    else:
        report.append("  • Scene complexity: Complex (many objects)")

    report.append("")
    report.append("="*80)

    return '\n'.join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_video.py <results.json>")
        print("\nExample:")
        print("  python analyze_video.py output/2025-07-20_15-11-29_results.json")
        sys.exit(1)

    json_path = sys.argv[1]

    print("Loading results...")
    results = load_results(json_path)

    print("Analyzing video...\n")
    report = generate_video_understanding(results)

    print(report)

    # Save report
    output_path = json_path.replace('.json', '_analysis.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 Analysis saved to: {output_path}")


if __name__ == '__main__':
    main()
