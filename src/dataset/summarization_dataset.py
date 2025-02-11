"""
Usage: `python3 -m dataset.summarization_dataset --seed 42 --datasets pubmed qmsum govreport arxiv
--min_doc_length 1500 --max_doc_length 15000 --min_summary_length 75
--max_summary_length 750 --num_records 200`
"""

import argparse
import random
import pandas as pd
from abc import ABC, abstractmethod
from datasets import load_dataset

from resources.utils import seed_everything


class AbstractDatasetProcessor(ABC):
    """Abstract base class for dataset processing."""

    def __init__(
        self,
        min_doc_length,
        max_doc_length,
        min_summary_length,
        max_summary_length,
        num_records=200,
    ):
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.min_summary_length = min_summary_length
        self.max_summary_length = max_summary_length
        self.num_records = num_records

    @abstractmethod
    def load_dataset(self):
        """Load the dataset."""
        raise NotImplementedError

    @abstractmethod
    def extract_document_summary(self, record):
        """Extract document and summary from a record."""
        raise NotImplementedError

    @abstractmethod
    def save_to_csv(self, documents, summaries, sources, filename):
        """Save the documents, summaries, and sources to a CSV file."""
        raise NotImplementedError

    def filter_records(self, dataset):
        """Filter the records based on length constraints."""
        documents = []
        summaries = []
        for record in dataset:
            document, summary = self.extract_document_summary(record)
            if (
                self.min_doc_length <= len(document.split()) <= self.max_doc_length
                and self.min_summary_length
                <= len(summary.split())
                <= self.max_summary_length
            ):
                documents.append(document)
                summaries.append(summary)
        return documents, summaries

    def sample_records(self, documents, summaries, source_label):
        """Sample records from the filtered documents and summaries."""
        indices = random.sample(range(len(documents)), self.num_records)
        indices.sort()

        sampled_documents = [documents[i] for i in indices]
        sampled_summaries = [summaries[i] for i in indices]
        sampled_sources = [source_label] * self.num_records

        return sampled_documents, sampled_summaries, sampled_sources

    def process(self, source_label, filename):
        """Process the dataset and save the results to a CSV file."""
        dataset = self.load_dataset()
        documents, summaries = self.filter_records(dataset)
        sampled_documents, sampled_summaries, sampled_sources = self.sample_records(
            documents, summaries, source_label
        )
        self.save_to_csv(
            sampled_documents, sampled_summaries, sampled_sources, filename
        )


class PubMedProcessor(AbstractDatasetProcessor):
    """Processor for PubMed dataset."""

    def load_dataset(self):
        return load_dataset("ccdv/pubmed-summarization", split="train")

    def extract_document_summary(self, record):
        document = record["article"]
        summary = record["abstract"]
        return document, summary

    def save_to_csv(self, documents, summaries, sources, filename):
        data = {"document": documents, "summary": summaries, "source": sources}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(filename, index=False)


class QMSumProcessor(AbstractDatasetProcessor):
    """Processor for QMSum dataset."""

    def load_dataset(self):
        return load_dataset("pszemraj/qmsum-cleaned", split="train")

    def extract_document_summary(self, record):
        document = record["input"]
        summary = record["output"]
        return document, summary

    def save_to_csv(self, documents, summaries, sources, filename):
        data = {"document": documents, "summary": summaries, "source": sources}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(filename, index=False)


class GovReportProcessor(AbstractDatasetProcessor):
    """Processor for GovReport dataset."""

    def load_dataset(self):
        return load_dataset("ccdv/govreport-summarization", split="train")

    def extract_document_summary(self, record):
        document = record["report"]
        summary = record["summary"]
        return document, summary

    def save_to_csv(self, documents, summaries, sources, filename):
        data = {"document": documents, "summary": summaries, "source": sources}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(filename, index=False, escapechar="\\")


class ArxivProcessor(AbstractDatasetProcessor):
    """Processor for Arxiv dataset."""

    def load_dataset(self):
        arxiv = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
        arxiv_subset = arxiv.take(2000)
        return list(arxiv_subset)

    def extract_document_summary(self, record):
        document = record["article"]
        summary = record["abstract"]
        return document, summary

    def save_to_csv(self, documents, summaries, sources, filename):
        data = {"document": documents, "summary": summaries, "source": sources}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(filename, index=False)


class SummScreenProcessor(AbstractDatasetProcessor):
    """Processor for SummScreen dataset."""

    def load_dataset(self):
        return load_dataset("YuanPJ/summ_screen", "all", split="train")

    def extract_document_summary(self, record):
        document = "\n".join(record["Transcript"]).replace("(", " ").replace(")", " ")
        summary = record["Recap"][0]
        return document, summary

    def save_to_csv(self, documents, summaries, sources, filename):
        data = {"document": documents, "summary": summaries, "source": sources}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(filename, index=False)


class BigPatentProcessor(AbstractDatasetProcessor):
    """Processor for BigPatent dataset."""

    def load_dataset(self):
        subsets = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]
        all_records = []
        for subset in subsets:
            dataset = load_dataset("big_patent", subset, split="train", streaming=True)
            dataset = dataset.take(50)
            all_records.extend(list(dataset))
        return all_records

    def extract_document_summary(self, record):
        document = record["description"]
        summary = record["abstract"]
        return document, summary

    def save_to_csv(self, documents, summaries, sources, filename):
        data = {"document": documents, "summary": summaries, "source": sources}
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(filename, index=False)


def main(
    datasets,
    min_doc_length,
    max_doc_length,
    min_summary_length,
    max_summary_length,
    num_records,
):
    """Main function to process datasets based on provided parameters."""
    processor_classes = {
        "pubmed": PubMedProcessor,
        "qmsum": QMSumProcessor,
        "govreport": GovReportProcessor,
        "arxiv": ArxivProcessor,
        "summ_screen": SummScreenProcessor,
        "big_patent": BigPatentProcessor,
    }

    for dataset_name in datasets:
        if dataset_name in processor_classes:
            processor = processor_classes[dataset_name](
                min_doc_length,
                max_doc_length,
                min_summary_length,
                max_summary_length,
                num_records,
            )
            processor.process(source_label=dataset_name, filename=f"prepared_dataset/{dataset_name}.csv")
        else:
            print(f"Dataset {dataset_name} is not supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and sample datasets.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True, help="List of datasets to process",
        choices=["pubmed", "qmsum", "govreport", "arxiv", "summ_screen", "big_patent"]
    )
    parser.add_argument(
        "--min_doc_length", type=int, default=1500, help="Minimum document length"
    )
    parser.add_argument(
        "--max_doc_length", type=int, default=15000, help="Maximum document length"
    )
    parser.add_argument(
        "--min_summary_length", type=int, default=75, help="Minimum summary length"
    )
    parser.add_argument(
        "--max_summary_length", type=int, default=750, help="Maximum summary length"
    )
    parser.add_argument(
        "--num_records", type=int, default=200, help="Number of records to sample"
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    main(
        datasets=args.datasets,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        min_summary_length=args.min_summary_length,
        max_summary_length=args.max_summary_length,
        num_records=args.num_records,
    )
