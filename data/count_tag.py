train_positive, train_negative, train_neutral = 0, 0, 0
val_positive, val_negative, val_neutral = 0, 0, 0
positive, negative, neutral = 0, 0, 0
with open('data/train.txt', 'r', encoding='utf-8') as f:
    f.readline()
    lines = f.readlines()
    for line in lines:
        tag = line.strip().split(',')[1]
        if tag == 'positive':
            train_positive += 1
            positive += 1
        elif tag == 'negative':
            train_negative += 1
            negative += 1
        elif tag == 'neutral':
            train_neutral += 1
            neutral += 1
with open('data/val.txt', 'r', encoding='utf-8') as f:
    f.readline()
    lines = f.readlines()
    for line in lines:
        tag = line.strip().split(',')[1]
        if tag == 'positive':
            val_positive += 1
            positive += 1
        elif tag == 'negative':
            val_negative += 1
            negative += 1
        elif tag == 'neutral':
            val_neutral += 1
            neutral += 1
print(f'train_positive: {train_positive} ({train_positive/3500:.2%})')
print(f'train_negative: {train_negative} ({train_negative/3500:.2%})')
print(f'train_neutral: {train_neutral} ({train_neutral/3500:.2%})')

print(f'val_positive: {val_positive} ({val_positive/500:.2%})')
print(f'val_negative: {val_negative} ({val_negative/500:.2%})')
print(f'val_neutral: {val_neutral} ({val_neutral/500:.2%})')

assert train_positive + train_negative + train_neutral == 3500
assert val_positive + val_negative + val_neutral == 500

print(f'total_positive: {positive} ({positive/4000:.2%})')
print(f'total_negative: {negative} ({negative/4000:.2%})')
print(f'total_neutral: {neutral} ({neutral/4000:.2%})')
    