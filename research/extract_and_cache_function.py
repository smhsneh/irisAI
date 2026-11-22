def extract_and_cache(split_name, jsonl_file):
    split_cache = os.path.join(CACHE_DIR, split_name)
    os.makedirs(split_cache, exist_ok=True)

    with open(jsonl_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    model.eval()
    print(f"\nExtracting {split_name}: {len(records)} samples")

    for start in tqdm(range(0, len(records), SHARD_SIZE)):
        end = min(start + SHARD_SIZE, len(records))

        shard_path = os.path.join(
            split_cache, f"part_{start // SHARD_SIZE:04d}.pt"
        )

        if os.path.exists(shard_path):
            print("Skipping existing shard:", shard_path)
            continue

        shard_records = records[start:end]

        shard_data = {
            "image_pooled": [],
            "text_pooled": [],
            "image_tokens": [],
            "text_tokens": [],
            "attention_mask": [],
            "input_ids": [],
            "labels": [],
            "ids": [],
            "texts": []
        }

        for batch_start in range(0, len(shard_records), BATCH_SIZE):
            batch_records = shard_records[
                batch_start:batch_start + BATCH_SIZE
            ]

            images = []

            for record in batch_records:
                image_path = os.path.join(DATA_DIR, record["img"])

                if not os.path.isfile(image_path):
                    raise FileNotFoundError(image_path)

                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))

            texts = [record["text"] for record in batch_records]

            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            )

            inputs = {
                key: value.to("cuda")
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    outputs = model(**inputs, return_dict=True)

            shard_data["image_pooled"].append(
                outputs.image_embeds.cpu().half()
            )
            shard_data["text_pooled"].append(
                outputs.text_embeds.cpu().half()
            )
            shard_data["image_tokens"].append(
                outputs.vision_model_output.last_hidden_state.cpu().half()
            )
            shard_data["text_tokens"].append(
                outputs.text_model_output.last_hidden_state.cpu().half()
            )
            shard_data["attention_mask"].append(
                inputs["attention_mask"].cpu().bool()
            )
            shard_data["input_ids"].append(
                inputs["input_ids"].cpu()
            )

            shard_data["labels"].extend(
                [int(record["label"]) for record in batch_records]
            )
            shard_data["ids"].extend(
                [record.get("id") for record in batch_records]
            )
            shard_data["texts"].extend(texts)

        for key in [
            "image_pooled", "text_pooled",
            "image_tokens", "text_tokens",
            "attention_mask", "input_ids"
        ]:
            shard_data[key] = torch.cat(shard_data[key], dim=0)

        shard_data["labels"] = torch.tensor(
            shard_data["labels"], dtype=torch.long
        )

        torch.save(shard_data, shard_path)

        del shard_data
        torch.cuda.empty_cache()

    print(f"{split_name} extraction completed!")
