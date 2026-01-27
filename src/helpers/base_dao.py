from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple

from bson.objectid import ObjectId
from pymongo.collection import Collection
from pymongo.database import Database

Sort = Iterable[Tuple[str, int]]

@dataclass(slots=True)
class BaseDao:
    """CRUD helper for Mongo collections."""

    db: Database
    _hide_mongo_id: bool = False

    collection_name: ClassVar[str] = ""

    # -- Collection ---------------------------------------------------------
    @property
    def col(self) -> Collection:
        if not self.collection_name:
            msg = "collection_name must be set on subclass"
            raise ValueError(msg)
        return self.db[self.collection_name]

    @property
    def default_projection(self) -> Dict[str, int]:
        return {"_id": 0} if self._hide_mongo_id else {}

    # -- Read ---------------------------------------------------------------
    def find(
        self,
        query: Dict[str, Any] | None = None,
        *,
        projection: Dict[str, int] | None = None,
        sort: Sort | None = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        q = query or {}
        proj = projection or self.default_projection
        cursor = self.col.find(q, proj)
        if sort:
            cursor = cursor.sort(list(sort))
        if skip:
            cursor = cursor.skip(skip)
        if limit is not None:
            cursor = cursor.limit(int(limit))
        return self.serialize(list(cursor))

    def find_one(
        self,
        query: Dict[str, Any],
        *,
        projection: Dict[str, int] | None = None,
    ) -> Dict[str, Any] | None:
        return self.serialize(self.col.find_one(query, projection or self.default_projection))

    def count(self, query: Dict[str, Any] | None = None) -> int:
        return self.col.count_documents(query or {})

    def paginate(
        self,
        query: Dict[str, Any] | None,
        *,
        page: int = 1,
        per_page: int = 20,
        sort: Sort | None = None,
        projection: Dict[str, int] | None = None,
    ) -> Dict[str, Any]:
        total = self.count(query or {})
        items = self.find(
            query or {},
            projection=projection,
            sort=sort,
            limit=per_page,
            skip=(page - 1) * per_page,
        )
        return {"items": items, "page": page, "per_page": per_page, "total": total}

    # -- Write --------------------------------------------------------------
    def insert_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.col.insert_one(payload)
        return self.serialize(payload)

    def insert_many(self, payloads: List[Dict[str, Any]]) -> int:
        return len(self.col.insert_many(payloads).inserted_ids)

    def update_one(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
        *,
        upsert: bool = False,
        set_operator: bool = True,
    ) -> int:
        ops = {"$set": update} if set_operator else update
        res = self.col.update_one(query, ops, upsert=upsert)
        return res.modified_count + (1 if res.upserted_id else 0)

    def delete_one(self, query: Dict[str, Any]) -> int:
        return self.col.delete_one(query).deleted_count
    
    def delete_many(self, query: Dict[str, Any]) -> int:
        return self.col.delete_many(query).deleted_count

    # -- Utils --------------------------------------------------------------
    def serialize(self, response: Any) -> Any:
        if isinstance(response, ObjectId):
            return str(response)
        if isinstance(response, dict):
            return {key: self.serialize(value) for key, value in response.items()}
        if isinstance(response, list):
            return [self.serialize(item) for item in response]
        return response

