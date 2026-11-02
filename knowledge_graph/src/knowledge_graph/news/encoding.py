"""Long-input encoding shared by ATLOP and DREEAM."""

import torch
import torch.nn.functional as F


def encode_layers(model, input_ids, attention_mask, *, layers):
    """Last-layer states and attention, or the mean of the last ``layers`` layers."""
    if layers == 1:
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        return output[0], output[-1][-1]
    output = model(
        input_ids=input_ids, attention_mask=attention_mask,
        output_attentions=True, output_hidden_states=True,
    )
    return torch.stack(output[-2][-layers:], dim=1).mean(dim=1), torch.stack(output[-1][-layers:], dim=1).mean(dim=1)


def boundary_tokens(config):
    end_tokens = [config.sep_token_id] * (2 if config.transformer_type == "roberta" else 1)
    return [config.cls_token_id], end_tokens


def encode_chunks(model, input_ids, attention_mask, *, layers):
    """Encode up to 1,024 subwords; add a central chunk if the first and last chunks leave a gap."""
    padded_length = input_ids.size(1)
    start_tokens, end_tokens = boundary_tokens(model.config)
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    start_length = start_tokens.size(0)
    end_length = end_tokens.size(0)
    if padded_length <= 512:
        return encode_layers(model, input_ids, attention_mask, layers=layers)

    new_input_ids, new_attention_mask, chunk_counts = [], [], []
    lengths = attention_mask.sum(1).long().tolist()
    for row, length in enumerate(lengths):
        if length <= 512:
            new_input_ids.append(input_ids[row, :512])
            new_attention_mask.append(attention_mask[row, :512])
            chunk_counts.append(1)
        else:
            input_ids1 = torch.cat([input_ids[row, : 512 - end_length], end_tokens], dim=-1)
            input_ids2 = torch.cat([start_tokens, input_ids[row, (length - 512 + start_length) : length]], dim=-1)
            attention_mask1 = attention_mask[row, :512]
            attention_mask2 = attention_mask[row, (length - 512) : length]
            new_input_ids.extend([input_ids1, input_ids2])
            new_attention_mask.extend([attention_mask1, attention_mask2])
            chunk_counts.append(2)
            if 512 - end_length < length - 512 + start_length:
                body_length = 512 - start_length - end_length
                start = (length - body_length) // 2
                middle = torch.cat([start_tokens, input_ids[row, start : start + body_length], end_tokens])
                new_input_ids.append(middle)
                new_attention_mask.append(torch.ones_like(attention_mask1))
                chunk_counts[-1] = 3
    input_ids = torch.stack(new_input_ids, dim=0)
    attention_mask = torch.stack(new_attention_mask, dim=0)
    sequence_output, attention = encode_layers(model, input_ids, attention_mask, layers=layers)
    chunk_start = 0
    new_output, new_attention = [], []
    for chunk_count, length in zip(chunk_counts, lengths):
        if chunk_count == 1:
            output = F.pad(sequence_output[chunk_start], (0, 0, 0, padded_length - 512))
            att = F.pad(attention[chunk_start], (0, padded_length - 512, 0, padded_length - 512))
        else:
            output1 = sequence_output[chunk_start][: 512 - end_length]
            mask1 = attention_mask[chunk_start][: 512 - end_length]
            att1 = attention[chunk_start][:, : 512 - end_length, : 512 - end_length]
            output1 = F.pad(output1, (0, 0, 0, padded_length - 512 + end_length))
            mask1 = F.pad(mask1, (0, padded_length - 512 + end_length))
            att1 = F.pad(att1, (0, padded_length - 512 + end_length, 0, padded_length - 512 + end_length))

            output2 = sequence_output[chunk_start + 1][start_length:]
            mask2 = attention_mask[chunk_start + 1][start_length:]
            att2 = attention[chunk_start + 1][:, start_length:, start_length:]
            output2 = F.pad(output2, (0, 0, length - 512 + start_length, padded_length - length))
            mask2 = F.pad(mask2, (length - 512 + start_length, padded_length - length))
            att2 = F.pad(att2, [
                length - 512 + start_length, padded_length - length,
                length - 512 + start_length, padded_length - length,
            ])
            mask = mask1 + mask2
            output = output1 + output2
            att = att1 + att2
            if chunk_count == 3:
                body_length = 512 - start_length - end_length
                start = (length - body_length) // 2
                stop = start + body_length
                gap_start, gap_stop = 512 - end_length, length - 512 + start_length
                rows = slice(start_length + gap_start - start, start_length + gap_stop - start)
                columns = slice(start_length, 512 - end_length)
                # The central chunk supplies only the rows missing from the outer chunks.
                output = output + F.pad(
                    sequence_output[chunk_start + 2, rows], (0, 0, gap_start, padded_length - gap_stop),
                )
                mask = mask + F.pad(attention_mask[chunk_start + 2, rows], (gap_start, padded_length - gap_stop))
                att = att + F.pad(
                    attention[chunk_start + 2, :, rows, columns],
                    (start, padded_length - stop, gap_start, padded_length - gap_stop),
                )
            output = output / (mask + 1e-10).unsqueeze(-1)
            att = att / (att.sum(-1, keepdim=True) + 1e-10)
        new_output.append(output)
        new_attention.append(att)
        chunk_start += chunk_count
    return torch.stack(new_output, dim=0), torch.stack(new_attention, dim=0)


def pool_entities(sequence, attention, entities, offset):
    """Log-sum-exp mention states and average their attention within each entity."""
    embeddings, attentions = [], []
    heads, _, length = attention.shape
    for mentions in entities:
        positions = [start + offset for start, _ in mentions if start + offset < length]
        if len(positions) > 1:
            embedding = torch.logsumexp(torch.stack([sequence[i] for i in positions]), dim=0)
            weights = torch.stack([attention[:, i] for i in positions]).mean(0)
        elif positions:
            embedding, weights = sequence[positions[0]], attention[:, positions[0]]
        else:
            embedding = sequence.new_zeros(sequence.shape[-1])
            weights = attention.new_zeros(heads, length)
        embeddings.append(embedding)
        attentions.append(weights)
    return torch.stack(embeddings), torch.stack(attentions)


def encode_with_windows(model, input_ids, attention_mask, *, layers: int):
    """Average half-overlapping 512-token windows and normalize merged attention rows."""
    batch, padded = input_ids.size()
    start_tokens, end_tokens = boundary_tokens(model.config)
    start_tokens = torch.tensor(start_tokens, device=input_ids.device, dtype=input_ids.dtype)
    end_tokens = torch.tensor(end_tokens, device=input_ids.device, dtype=input_ids.dtype)
    len_start, len_end = start_tokens.numel(), end_tokens.numel()
    max_body = 512 - len_start - len_end
    stride = max_body // 2
    if padded <= 512:
        return encode_layers(model, input_ids, attention_mask, layers=layers)

    sequence_outputs, attentions = [], []
    for row in range(batch):
        length = int(attention_mask[row].sum())
        if length <= 512:
            hidden, attention = encode_layers(
                model, input_ids[row : row + 1, :length], attention_mask[row : row + 1, :length],
                layers=layers,
            )
            sequence_outputs.append(F.pad(hidden[0], (0, 0, 0, padded - length)))
            attentions.append(F.pad(attention[0], (0, padded - length, 0, padded - length)))
            continue
        # Single-sequence inputs have one CLS and one SEP, even when a window adds two SEP tokens.
        body_length = length - 2
        starts = list(range(0, max(1, body_length - max_body + 1), stride))
        final_start = max(0, body_length - max_body)
        if starts[-1] != final_start:
            starts.append(final_start)

        windows, masks, spans = [], [], []
        for start in starts:
            end = min(body_length, start + max_body)
            body = input_ids[row, len_start + start : len_start + end]
            window = torch.cat([start_tokens, body, end_tokens])
            mask = torch.ones_like(window)
            pad = 512 - window.numel()
            if pad:
                window = F.pad(window, (0, pad))
                mask = F.pad(mask, (0, pad))
            windows.append(window)
            masks.append(mask)
            spans.append((start, end))

        window_output, window_attention = encode_layers(model, torch.stack(windows), torch.stack(masks), layers=layers)
        dimension, heads = window_output.size(-1), window_attention.size(1)
        output_sum = torch.zeros((padded, dimension), device=input_ids.device, dtype=torch.float32)
        output_count = torch.zeros((padded, 1), device=input_ids.device, dtype=torch.float32)
        attention_sum = torch.zeros((heads, padded, padded), device=input_ids.device, dtype=torch.float32)
        attention_count = torch.zeros((padded, padded), device=input_ids.device, dtype=torch.float32)
        for index, (start, end) in enumerate(spans):
            local_length = end - start
            target = slice(len_start + start, len_start + end)
            source = slice(len_start, len_start + local_length)
            output_sum[target] += window_output[index, source].float()
            output_count[target] += 1
            attention_sum[:, target, target] += window_attention[index, :, source, source].float()
            attention_count[target, target] += 1

        sequence_outputs.append((output_sum / output_count.clamp_min(1)).to(window_output.dtype))
        attention = attention_sum / attention_count.clamp_min(1).unsqueeze(0)
        attention = attention / (attention.sum(-1, keepdim=True) + 1e-10)
        attentions.append(attention.to(window_attention.dtype))
    return torch.stack(sequence_outputs), torch.stack(attentions)
